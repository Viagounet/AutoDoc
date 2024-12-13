import glob
import random
from copy import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from modules.rouge_metric import RougeCalculator

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

try:
	_model = AutoModelForSequenceClassification.from_pretrained(
		"./output-experiments/dialogue-act-model/bert-base-uncased-silicone")
	_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
	pipe = pipeline(
		'text-classification',
		model=_model,
		tokenizer=_tokenizer,
		return_all_scores=True,
	)
except:
	pipe = None

pipe = None


########################################################################################
########################################################################################
########################################################################################
class Node:
	def __init__(self, num, history):
		self.num = num
		self.history = history


import re


def extract_bullet_points(text: str) -> list:
	bullet_point_formats = r"[-•*?●○■]"
	lines = text.split("\n")
	bullet_points = []

	for line in lines:
		line = line.strip()
		if re.match(bullet_point_formats, line):
			cleaned_line = re.sub(bullet_point_formats, "", line).strip()
			no_brackets = re.sub(r'\[.*?\],?', '', cleaned_line)
			words = no_brackets.split()
			if len(words) >= 3:
				bullet_points.append(cleaned_line)

	return bullet_points


def create_nfn(num: int, pred: Node, stride: int):
	"""create_nfn (create node from node)

	Args:
		num (int): the id of the node we create
		pred (Node): The predecessor of the created node
		stride (int): _description_

	Returns:
		Node: return the created node.
	"""
	res = Node(num, None)

	temp = pred.history
	temp.append(num)  # add the current node to the history

	if stride is not None and len(temp) > stride:
		temp = temp[-stride:]

	res.history = temp

	return res


def entailment_prob(
		model,
		tokenizer,
		premise: Node,
		hypothesis: Node,
		transcript,
):
	"""entailment_prob

	The goal of this function is to calculate the entailement probability
	between the a Node and a given history.

	Args:
		model : a bert type model trained on dial-nli task
		tokenizer : the tokenizer which corresponds to the previous model
		premise (Node): the node that comes before the hypothesis
		hypothesis (Node): the current node to add
		transcript: a transcript type object

	Returns:
		Float : probability of entailment between the premise and the hypothesis
	"""
	p = " ".join([transcript[i] for i in premise.history])
	h = transcript[hypothesis.num]
	input = tokenizer(p + " [SEP] " + h, return_tensors='pt')
	output = model(**input)
	prob = torch.softmax(output.logits, dim=-1)
	return prob[0].item()


########################################################################################
########################################################################################
########################################################################################


def find_id(file):
	regex_results = re.search(r".*_annot([\d]*).txt", file)
	align_id = "ORIG"
	if regex_results:
		align_id = f"GENER_annot{regex_results.group(1)}"
	return align_id


def alignment_find_id(file):
	file = file.split("+")[2]
	return find_id(file)


def mean_pooling(model_output, attention_mask):
	token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sbert_embeddings(tokenizer, model, sentences):
	# Tokenize sentences
	encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

	# Compute token embeddings
	with torch.no_grad():
		model_output = model(**encoded_input)

	# Perform pooling
	sentences = mean_pooling(model_output, encoded_input['attention_mask'])

	# Normalize embeddings
	sentences = F.normalize(sentences, p=2, dim=1)
	return sentences


class Conversation:
	def __init__(self, folder):
		self._folder = folder
		self._transcript = Transcript(glob.glob(f"{folder}/transcript*.txt")[0])
		self._real_summaries_files = {find_id(file): file for file in glob.glob(f"{folder}/minutes*.txt")}
		self._real_summaries = {id: RealSummary(file, id) for id, file in self._real_summaries_files.items()}
		self._alignment_files = {id: file for id, file in
								 zip([alignment_find_id(file) for file in glob.glob(f"{folder}/align*.txt")],
									 glob.glob(f"{folder}/align*.txt"))}
		self._alignments = {id: Alignment(file, id) for id, file in self._alignment_files.items()}
		for id, alignment in self._alignments.items():
			self._real_summaries[id].link_to(self._transcript, alignment)

	@property
	def summaries(self):
		return list(self._real_summaries.values())

	def intra_rouge_score(self):
		similarities = []
		for id, summary in self._real_summaries.items():
			for id2, summary2 in self._real_summaries.items():
				if id == id2:
					continue
				similarities.append(summary.compare_with_truth(summary2))
		return {"R1": sum([sim["rouge1_fmeasure"] for sim in similarities]) / len(similarities),
				"R2": sum([sim["rouge2_fmeasure"] for sim in similarities]) / len(similarities),
				"RL": sum([sim["rougeL_fmeasure"] for sim in similarities]) / len(similarities)}

	def evaluate_summary(self, summary, take_max, stop_words):
		similarities = []
		for id, real_summary in self._real_summaries.items():
			similarities.append(summary.compare_with_truth(real_summary, stop_words))
		if take_max:
			return {"R1": max([sim["rouge1_fmeasure"] for sim in similarities]),
					"R2": max([sim["rouge2_fmeasure"] for sim in similarities]),
					"RL": max([sim["rougeL_fmeasure"] for sim in similarities]),
					"R1_precision": max([sim["rouge1_precision"] for sim in similarities]),
					"R2_precision": max([sim["rouge2_precision"] for sim in similarities]),
					"RL_precision": max([sim["rougeL_precision"] for sim in similarities]),
					"R1_recall": max([sim["rouge1_recall"] for sim in similarities]),
					"R2_recall": max([sim["rouge2_recall"] for sim in similarities]),
					"RL_recall": max([sim["rougeL_recall"] for sim in similarities])
					}
		return {"R1": sum([sim["rouge1_fmeasure"] for sim in similarities]) / len(similarities),
				"R2": sum([sim["rouge2_fmeasure"] for sim in similarities]) / len(similarities),
				"RL": sum([sim["rougeL_fmeasure"] for sim in similarities]) / len(similarities),
				"R1_precision": sum([sim["rouge1_precision"] for sim in similarities]) / len(similarities),
				"R2_precision": sum([sim["rouge2_precision"] for sim in similarities]) / len(similarities),
				"RL_precision": sum([sim["rougeL_precision"] for sim in similarities]) / len(similarities),
				"R1_recall": sum([sim["rouge1_recall"] for sim in similarities]) / len(similarities),
				"R2_recall": sum([sim["rouge2_recall"] for sim in similarities]) / len(similarities),
				"RL_recall": sum([sim["rougeL_recall"] for sim in similarities]) / len(similarities)
				}

	def gen_summary(self, gen_function):
		gen_summary = gen_function(self._transcript)
		return gen_summary

	def save_summary(self, summary, folder, file_prefix):
		with open(f"{folder}/_{file_prefix}_minutes.txt", "w", encoding="utf-8") as f:
			f.write(summary.minutes)

	def save_simple_transcript(self, simplification_model):

		simplification_model = simplification_model.split("/")[-1]
		self.transcript.simple[simplification_model].save(f"{self.folder}/_{simplification_model}_transcript.txt")

	@property
	def transcript(self):
		return self._transcript

	def has_alignments(self):
		if self._alignments:
			return True
		return False

	def get_summary_by_annot(self, annot):
		return self._real_summaries[annot]

	def generate_custom_alignment(self, summary, threshold):
		tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
		from transformers import AutoModel
		model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
		bullet_points = extract_bullet_points(summary.minutes)
		minutes_embeddings = sbert_embeddings(tokenizer, model, bullet_points)
		transcript_embeddings = sbert_embeddings(tokenizer, model, [l.cleaned_content for l in self.transcript.lines])
		similarities = cosine_similarity(minutes_embeddings, transcript_embeddings)
		lines_mapping = []
		for i, minute_similarities in enumerate(similarities):
			above_threshold_indices = np.argwhere(minute_similarities > threshold)
			mapping_minute = {"minute_line": i, "minute_content": bullet_points[i],
							  "mapping": []}
			if len(above_threshold_indices) == 0:
				continue
			for j in above_threshold_indices:
				j = int(j[0])
				sim = round(float(minute_similarities[j]), 2)
				mapping_minute["mapping"].append((f"{self.transcript.lines[j].speaker}"
												  f": {self.transcript.lines[j].content}", sim))
			lines_mapping.append(mapping_minute)
		return lines_mapping

	@property
	def alignments(self):
		return list(self._alignments.values())

	@property
	def folder(self):
		return self._folder

	def __str__(self):
		return f"Conversation in {self._folder}" + "\n" + str(self._transcript)


dates = r"\b[dD]ate|[dD]eadline|[mM]onday|[tT]uesday|[wW]ednesday|[tT]hursday|[fF]riday|[mM]orning|[aA]fternoon|[tT]omorrow|[yY]esterday"
warnings = r"[wW]arning|[aA]ttention|[cC]areful|[dD]eadline"
discussions = r"discuss|talk"
waiting = r"still|wait"


def choose_emoji(sentence, is_topic=False):
	if is_topic:
		if re.search(r"Attendees", sentence) != None:
			sentence = '👥 ' + sentence
		elif re.search(r"Subbmitted", sentence) != None:
			sentence = '✍ ' + sentence
		else:
			sentence = '🔷 ' + sentence
	else:
		tab = '\t'
		if re.search(dates, sentence) != None:
			sentence = tab + '📅 ' + sentence
		elif re.search(discussions, sentence) != None:
			sentence = tab + '💬 ' + sentence
		else:
			sentence = tab + '🔹 ' + sentence
		# check warnings and completed tasks
		if re.search(warnings, sentence) != None:
			sentence = sentence + ' ⚠️'
		elif re.search(waiting, sentence) != None:
			sentence = sentence + ' ⌛'

	return sentence


class Summary:
	def __init__(self):
		self._minutes = {}
		self._attendees = []

	@property
	def topics(self):
		return list(self._minutes.keys())

	def find_attendees(self, transcript):
		self._attendees = transcript._attendees

	@property
	def minutes(self):
		string = f"Date: {str(random.randint(1, 29))}.{str(random.randint(1, 12))}.2022\n👥 Attendees: " \
				 f"{', '.join([person for person in self._attendees])}\n\n"
		for topic, points in self._minutes.items():
			string += "\n" + choose_emoji(topic, True)
			string+= "\n"
			for bullet_point in points:
				for sentence in bullet_point.split(". "):
					if sentence == "":
						string += "\n"
						continue
					string += choose_emoji(sentence)
					string += "\n"
		string += "\n✍ Submitted by: Tirthankar"
		return string

	def compare_with_truth(self, real_summary, stop_words):
		rouge = RougeCalculator(stopwords=stop_words, lang="en")
		scores_dictionnary = {}
		for n in [1, 2]:
			for alpha, measure_type in zip([0, 0.5, 1], ["recall", "fmeasure", "precision"]):
				scores_dictionnary[f"rouge{n}_{measure_type}"] = rouge.rouge_n(n=n, summary=self.minutes,
																			   references=real_summary.minutes,
																			   alpha=alpha)
		for alpha, measure_type in zip([0, 0.5, 1], ["recall", "fmeasure", "precision"]):
			scores_dictionnary[f"rougeL_{measure_type}"] = rouge.rouge_l(summary=self.minutes,
																		 references=real_summary.minutes,
																		 alpha=alpha)
		return scores_dictionnary

	def __str__(self):
		return self.minutes


class SummaryFromFile(Summary):
	def __init__(self, file):
		super().__init__()
		self._file = file
		self._embeddings = None
		self._embeddings_topics = None
		self._load()

	@property
	def minutes(self):
		return self._content

	def get_line(self, summary_line):
		return self._lines[summary_line - 1]

	@property
	def lines(self):
		return self._lines

	def _load(self):
		with open(self._file, "r", encoding="utf-8") as f:
			self._content = f.read()
			self._lines = self._content.split("\n")

	def __str__(self):
		return f"Summary from : {self._file}\n{self.minutes}"


class RealSummary(SummaryFromFile):
	def __init__(self, file, annot_id):
		super().__init__(file)
		self._annot_id = annot_id
		self._embeddings = None
		self._embeddings_topics = None
		self._mapping = {}
		self._linked_transcription = None
		self._linked = False
		self._linked_alignment = None

	def link_to(self, transcript, alignment):
		for i, line_content in enumerate(self._content.split("\n")):
			self._mapping[i + 1] = [transcript[transcript_line - 1] for transcript_line in
									alignment.min2transcript(i + 1)]
		self._linked_alignment = alignment
		self._linked = True

	@property
	def linked(self):
		return self._linked

	@property
	def annotation_id(self):
		return self._annot_id

	def linked_transcript_lines(self, minute_line):
		if not self._linked:
			raise AttributeError("Summary not linked with a transcription")
		return self._mapping[minute_line]


class GeneratedSummary(Summary):
	def __init__(self):
		super().__init__()
		self._embeddings = None
		self._embeddings_topics = None

	def new_topic(self, new_topic):
		if new_topic not in self.topics:
			self._minutes[new_topic] = []

	def add_minute(self, minute, topic):
		if topic not in self._minutes:
			raise KeyError(f"{topic} is not a valid topic. Valid topics : {self.topics}")
		self._minutes[topic].append(minute)


class TranscriptLine:
	def __init__(self, content, line_number, speaker, new_turn, th: float = 0.5):
		self._content = content
		self._line_number = line_number
		self._speaker = speaker
		self._embedding = None
		self._density = None
		self._new_turn = new_turn
		self.simple_content = "Not initialized yet"
		self.keep = True

		if pipe is not None:
			self.dialogue_act = -1
			scores = pipe(self.content)[0]
			for k in scores:
				if k['score'] >= th:
					self.dialogue_act = k['label']

	@property
	def tags(self):
		return list(set(re.findall(f'\[(.*?)]', self._content)))

	@property
	def content(self):
		return self._content

	def update_content(self, new_content):
		self._content = new_content

	@property
	def line_number(self):
		return self._line_number

	@property
	def new_turn(self):
		return self._new_turn

	@property
	def cleaned_content(self):
		content = self._content
		content = re.sub(r"<\w+\/>", "", content)
		content = re.sub(r"\(.*\)", "", content)
		return content

	@property
	def speaker(self):
		return self._speaker

	def set_embedding(self, embedding):
		self._embedding = embedding

	def __str__(self):
		return f"{self.line_number} - {self._content}"


class Transcript:
	def __init__(self, file):
		self._file = file
		self._folder = "/".join(self._file.split("/")[:-1])
		self._attendees = []
		with open(file, "r") as f:
			self._content = f.read()
			speaker = "Undefined"
			self._lines = []
			for i, line_content in enumerate(self._content.split("\n")):
				line_number = i + 1
				new_turn = False
				if re.search(r"\(PERSON[\d]*\)", line_content):
					speaker = "PERSON" + re.search(r"\(PERSON([\d]*)\)", line_content).group(1)
					self._attendees.append(speaker)
					new_turn = True
				self._lines.append(TranscriptLine(line_content, line_number, speaker, new_turn))
		self._attendees = list(set(self._attendees))
		if "simplification" not in self._file:
			self.simple = {f.split("_simplification_")[1]: Transcript(file=f) for f in
						   glob.glob(f"{self._folder}/_simplification*.txt")}

	@property
	def kept_lines(self):
		return [line for line in self._lines if line.keep]

	def simplify(self, model_path, tokenizer_path):
		import torch
		from transformers import AutoTokenizer
		from transformers import AutoModelForSeq2SeqLM

		tokenizer = AutoTokenizer.from_pretrained(model_path)
		model = AutoModelForSeq2SeqLM.from_pretrained(tokenizer_path)
		model_key = model_path.split("/")[-1].replace("transcript.txt", "")
		self.simple[model_key] = copy(self)
		self.simple[model_key]._file = None
		self.simple[model_key]._lines = []
		i = 0
		speaker = None
		for line in self.lines:
			input_ids = torch.tensor([tokenizer.encode(line.cleaned_content)])
			predict = model.generate(input_ids, max_length=512)[0]
			pred = tokenizer.decode(predict, skip_special_tokens=True)
			if pred in ["formula_13", "[PASS]", ""]:  # formula_13 is the returned sequence when no ids are passed
				pred = ""
				line.keep = False
			else:
				i += 1
				new_speaker = False
				if line.speaker != speaker:
					new_speaker = True
					speaker = line.speaker
				self.simple[model_key]._lines.append(TranscriptLine(pred, i, line.speaker, new_speaker))
			line.simple_content = pred
		return self.simple[model_key]

	def generate_embeddings(self, model, context_window=0, content="content"):
		relevant_lines = [getattr(line, content) for line in self.lines]
		if context_window:
			self._embeddings = model.encode(
				["\n".join(relevant_lines[i - context_window:i + context_window]) for i in range(len(self._lines))])
		else:
			self._embeddings = model.encode(relevant_lines)
		for embedding, line in zip(self._embeddings, self._lines):
			line.set_embedding(embedding)

	def save(self, output_file_path: str):
		sentences = []
		for line in self.lines:
			s = ""
			if line.new_turn:
				s += f"({line.speaker}) "
			s += line.content
			sentences.append(s)
		sentences = "\n".join(sentences)
		with open(output_file_path, "w") as f:
			f.write(sentences)

	@property
	def lines_strings(self):
		return [line.content for line in self._lines]

	@property
	def contains_project(self):
		return [line for line in self._lines if re.search(r"\[PROJECT[\d]*\]", line.content)]

	@property
	def projects(self):
		return "Not implemented yet"

	def window(self, line, size=[-3, 3]):
		size_min = size[0]
		size_max = size[1]
		if line.line_number - 1 + size_min < 0:
			size_min = -line.line_number + 1
		if line.line_number - 1 + size_max > len(self._lines):
			size_max = len(self._lines) - line.line_number

		return [line for line in self._lines[line.line_number - 1 + size_min:
											 line.line_number - 1 + size_max]]

	@property
	def content(self):
		return "\n".join(self.lines_strings)

	@property
	def lines(self):
		return self._lines

	def __getitem__(self, index):
		return self._lines[index]

	def __str__(self):
		string = f"Transcript file : {self._file}\n------------------\n" + \
				 "\n".join(self.lines_strings[:3]) + "\n...\n" + "\n".join(self.lines_strings[-3:])
		return string

	def __len__(self):
		return len(self._lines)

	def create_graph(self, model, tokenizer, stride):
		"""create_graph

		Args:
			model (_type_): _description_
			tokenizer (_type_): _description_
			stride (_type_): _description_
		"""

		dumb_root = Node(-1, None)
		first_node = Node(0, [0])

		conv_tree = [dumb_root, first_node]

		queue = list(range(len(self)))
		_ = queue.pop(0)  # on enlève au moins le premier

		path_list = [
			[-1, 0],  # the first path
		]  # list of all the paths in the conversational tree

		while queue:

			idx = queue.pop(0)
			candidat = Node(idx, None)

			buff = -100
			pred: Node = None

			# search for the best place to put our candidat
			for node in conv_tree:
				if node.num > 0:
					prob = entailment_prob(
						tokenizer,
						model,
						premise=node,
						hypothesis=candidat,
						transcript=self
					)
					if prob > buff:
						pred = node

			new_node = create_nfn(idx, pred, stride)
			conv_tree.append(new_node)

			# update the segmentation
			for i in range(len(path_list)):
				if path_list[i][-1] == pred.num:
					path_list[i].append(new_node.num)
					break

		self.dial_tree = path_list


class Alignment:
	def __init__(self, file_name, annot_id):
		self._annot_id = annot_id
		self._file = file_name
		self._df = pd.read_csv(file_name, sep=" ")
		self._df.columns = ['transcripts', 'minutes', 'label']

	@property
	def annotation_id(self):
		return self._annot_id

	def min2transcript(self, minute_line):
		return self._df[self._df["minutes"] == str(minute_line)]["transcripts"].tolist()

	def transcript2min(self, transcript_line):
		return self._df[self._df["transcripts"] == str(transcript_line)]["minutes"].tolist()

	@property
	def file(self):
		return self._file
