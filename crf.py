# -*- coding: utf-8 -*-
from conllu_parser import parse
from nltk.util import ngrams
import sklearn_crfsuite
import _pickle as pickle
from collections import OrderedDict
import math
import pprint as pp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

class FeatureExtr():
	'''
	Класс для извлечения признаков.
	'''

	def __init__(self):
		self.feature_list = ['word_is_upper', 'word_is_title', 'word_is_digit', 'pref[0]', 'suf[-1]', 'postag',
							 'pref[:2]', 'suf[-2:]', 'pref[:3]', 'suf[-3:]', 'pref[:4]', 'suf[-4:]']	# список признаков, общих для всех слов

	def load_conllu(self, filename):
		'''
		Загрузка файла в формате conllu и его парсинг.
		Для парсинга файлов с менее чем 10 колонками использовался этот парсер:
		https://github.com/svetlana21/conllu
		'''
		print('Loading starts...')
		with open(filename, 'r', encoding='utf-8') as f:
			data = f.read()
		result = parse(data, ['form','feats','misc'])
		print('Data is loaded.')
		return result

	def form_X(self, data):
		'''
		Дополнительное преобразование множества признаков - разделение грамматической информации на pos-теги и остальное
		:param data:
		:return:
		'''
		print('Changing of X starts...')
		X = []
		for sent in data:
			sent_X = []
			for word in sent:
				gram_list = word['feats'].split(',')	# разбиваем список грам.категорий на pos-тег и остальное
				if word['feats'][0] != ',':
					postag = gram_list[0]
					feats = gram_list[1:]
				else:
					postag = ','						# для ',' отдельное условие, т.к. по ней был split
					feats = []
				feats = OrderedDict([('form', word['form']), ('postag', postag), ('feats', feats), ('label', word['misc'])])
				sent_X.append(feats)
			X.append(sent_X)
		print('X is changed.')
		return X

	def gram(self, word):
		'''
		Преобразование грамматических категорий в словари. В дальнейшем будут использоваться как признаки.
		Списки грам.категорий отсюда: http://www.ruscorpora.ru/corpora-morph.html
		:param word: 
		:return: 
		'''
		cases = ['nom','gen','dat','acc','loc','ins','gen2','acc2','loc2','voc','adnum']
		genders = ['m','f','m-f','n']
		numbers = ['sg','pl']
		short_full_forms = ['brev','plen']
		voices = ['act','pass','med']
		verb_forms = ['inf','partcp','ger']
		postag = word['postag']
		new_feats = {}
		# падеж для сущ., прил. или числ.
		if postag == 'S' or postag == 'S-PRO' \
				or postag == 'A' or postag == 'A-NUM' or postag == 'A-PRO'\
				or postag == 'NUM':
			for case in cases:
				if case in word['feats']:
					new_feats.update({'case': case})
					break
		# род и число для сущ., прил. и гл.
		if postag == 'S' or postag == 'S-PRO' \
				or postag == 'A' or postag == 'A-NUM' or postag == 'A-PRO'\
				or postag == 'V':
			for gender in genders:
				if gender in word['feats']:
					new_feats.update({'gender': gender})
					break
			for number in numbers:
				if number in word['feats']:
					new_feats.update({'number': number})
					break
		# полные или краткие формы для прилагательных и глаголов
		if postag == 'A' or postag == 'A-NUM' or postag == 'A-PRO' \
				or postag == 'V':
			for form in short_full_forms:
				if form in word['feats']:
					new_feats.update({'short_full_form': form})
					break
		# залог и форма для глаголов
		if postag == 'V':
			for voice in voices:
				if voice in word['feats']:
					new_feats.update({'voice': voice})
					break
			for form in verb_forms:
				if form in word['feats']:
					new_feats.update({'verb_form': form})
					break
		return new_feats

	def all_words_features(self, word):
		'''
		Получение признаков, которые нужно извлечь для любого слова, вне зависимости от его положения в окне.
		Это такие признаки, как:
			1) токен в uppercase (или нет);
            2) токен с большой буквы (или нет);
            3) токен - это цифра (или нет);
            4) первая и последняя буквы;
            5) pos-тег;
            6) если длина слова > 1, то префиксы и суффиксы длины от 2 до 4 символов.
		'''
		word_features = [word['form'].isupper(), word['form'].istitle(), word['form'].isdigit(), word['form'][0], word['form'][-1], word['postag']]  # признаки 1-4
		if len(word) > 1:  # префиксы и суффиксы в зависимости от длины слова
			word_features.extend([word['form'][:2], word['form'][-2:]])
		if len(word) > 2:
			word_features.extend([word['form'][:3], word['form'][-3:]])
		if len(word) > 3:
			word_features.extend([word['form'][:4], word['form'][-4:]])
		return word_features

	def make_right_context_features(self, sent, i):
		'''
		Функция, формирующая признаки правого контекста текущего токена. Контекст +3 слова.
		Это те признаки, которые извлекаются с помощью функции all_words_features.
		Формат для признаков контекста - '+1/2/3:<признак>': '<значение>'.
		Функция также формирует список слов в правом контексте. 
		В дальнейшем это требуется для формирования списка всех слов окна, который передаётся в функцию ngrams.
		'''
		word1 = sent[i + 1]								# 1-е справа слово
		word1_feat = self.all_words_features(word1)		# признаки для 1-го справа слова
		r_context = [word1['form']]						# добавляем его в список правого контекста
		feature_list_plus_1 = ['+1:{}'.format(feat) for feat in self.feature_list]	# изменяем общий список признаков, добавляя к каждому +1
		r_context_features = dict(zip(feature_list_plus_1, word1_feat))		# словарь признаков

		feats = self.gram(word1)		# добавляем в качестве признаков грам. категории
		new_feats = {}
		for key in feats.keys():
			new_key = '+1:{}'.format(key)
			new_feats.update({new_key: feats[key]})
		r_context_features.update(new_feats)

		if i == len(sent) - 3:		# то же для 2-го справа слова (если оно есть)
			word2 = sent[i + 2]
			word2_feat = (self.all_words_features(word2))
			r_context.append(word2['form'])
			feature_list_plus_2 = ['+2:{}'.format(feat) for feat in self.feature_list]
			w2_features = dict(zip(feature_list_plus_2, word2_feat))
			r_context_features.update(w2_features)

			feats = self.gram(word2)
			new_feats = {}
			for key in feats.keys():
				new_key = '+2:{}'.format(key)
				new_feats.update({new_key: feats[key]})
			r_context_features.update(new_feats)

		if i < len(sent) - 3:		# то же для 3-го справа слова (если оно есть)
			word2 = sent[i + 2]
			word2_feat = (self.all_words_features(word2))
			word3 = sent[i + 3]
			word3_feat = (self.all_words_features(word3))
			r_context.extend([word2['form'], word3['form']])
			feature_list_plus_2 = ['+2:{}'.format(feat) for feat in self.feature_list]
			feature_list_plus_3 = ['+3:{}'.format(feat) for feat in self.feature_list]
			w2_features = dict(zip(feature_list_plus_2, word2_feat))
			w3_features = dict(zip(feature_list_plus_3, word3_feat))
			r_context_features.update(w2_features)
			r_context_features.update(w3_features)

			new_feats = {}
			feats2 = self.gram(word2)
			for key in feats2.keys():
				new_key = '+2:{}'.format(key)
				new_feats.update({new_key: feats2[key]})

			feats3 = self.gram(word3)
			for key in feats3.keys():
				new_key = '+3:{}'.format(key)
				new_feats.update({new_key: feats3[key]})
			r_context_features.update(new_feats)
		return r_context_features, r_context

	def make_left_context_features(self, sent, i):
		'''
		То же, что make_right_context_features, только для левого контекста (-3 слова).
		'''
		word1 = sent[i - 1]
		word1_feat = self.all_words_features(word1)
		l_context = [word1['form']]
		feature_list_minus_1 = ['-1:{}'.format(feat) for feat in self.feature_list]
		l_context_features = dict(zip(feature_list_minus_1, word1_feat))

		feats = self.gram(word1)
		new_feats = {}
		for key in feats.keys():
			new_key = '-1:{}'.format(key)
			new_feats.update({new_key: feats[key]})
		l_context_features.update(new_feats)

		if i == 2:
			word2 = sent[i - 2]
			word2_feat = self.all_words_features(word2)
			l_context.insert(0, word2['form'])
			feature_list_minus_2 = ['-2:{}'.format(feat) for feat in self.feature_list]
			w2_features = dict(zip(feature_list_minus_2, word2_feat))
			l_context_features.update(w2_features)

			feats = self.gram(word2)
			new_feats = {}
			for key in feats.keys():
				new_key = '-2:{}'.format(key)
				new_feats.update({new_key: feats[key]})
			l_context_features.update(new_feats)

		if i > 2:
			word2 = sent[i - 2]
			word2_feat = self.all_words_features(word2)
			word3 = sent[i - 3]
			word3_feat = self.all_words_features(word3)
			l_context.insert(0, word2['form'])
			l_context.insert(0, word3['form'])
			feature_list_minus_2 = ['-2:{}'.format(feat) for feat in self.feature_list]
			feature_list_minus_3 = ['-3:{}'.format(feat) for feat in self.feature_list]
			w2_features = dict(zip(feature_list_minus_2, word2_feat))
			w3_features = dict(zip(feature_list_minus_3, word3_feat))
			l_context_features.update(w2_features)
			l_context_features.update(w3_features)

			new_feats = {}
			feats2 = self.gram(word2)
			for key in feats2.keys():
				new_key = '-2:{}'.format(key)
				new_feats.update({new_key: feats2[key]})

			feats3 = self.gram(word3)
			for key in feats3.keys():
				new_key = '-3:{}'.format(key)
				new_feats.update({new_key: feats3[key]})
			l_context_features.update(new_feats)
		return l_context_features, l_context

	def ngrams(self, window):
		'''
		Признаки-биграммы и признаки-триграммы.
		'''
		ngrams_2 = list(ngrams(window, 2))
		ngrams_3 = list(ngrams(window, 3))
		bigrams = ['{} {}'.format(ngrams_2[i][0], ngrams_2[i][1]) for i in range(len(ngrams_2))]
		trigrams = ['{} {} {}'.format(ngrams_3[i][0], ngrams_3[i][1], ngrams_3[i][2]) for i in range(len(ngrams_3))]
		bigr_features = dict(zip(['bi_1', 'bi_2', 'bi_3', 'bi_4', 'bi_5', 'bi_6'], bigrams))
		trigr_features = dict(zip(['tri_1', 'tri_2', 'tri_3', 'tri_4', 'tri_5'], trigrams))
		return bigr_features, trigr_features

	def word2features(self, sent, i):
		'''
		Функция, формирующая полный список признаков:
			1) токен в uppercase (или нет);
			2) токен с большой буквы (или нет);
			3) токен - это цифра (или нет);
			4) первая и последняя буквы;
			5) pos-тег;
			6) если длина слова > 1, то префиксы и суффиксы длины от 2 до 4 символов;
			7) всё вышеперечисленное для правого контекста;
			8) всё вышеперечисленное для левого контекста;
			9) сам токен в lowercase;
			10) bias = 1.0;
			11) если токен является началом предложения, 'BOS' = True;
			12) если токен является началом предложения, 'EOS' = True;
			13) биграммы;
			14) триграммы.
		'''
		word = sent[i]
		#print(word['form'])
		word_feat = self.all_words_features(word)
		features = dict(zip(self.feature_list, word_feat))
		features.update({'word': word['form'].lower(),
						 'bias': 1.0})
		if len(sent) > 1:
			if i == 0:
				features['BOS'] = True
				right_context = self.make_right_context_features(sent, i)
				features.update(right_context[0])
				window = right_context[1]
				window.insert(0, word['form'])
			elif i == len(sent) - 1:
				features['EOS'] = True
				left_context = self.make_left_context_features(sent, i)
				features.update(left_context[0])
				window = left_context[1]
				window.append(word['form'])
			else:
				left_context = self.make_left_context_features(sent, i)
				features.update(left_context[0])
				right_context = self.make_right_context_features(sent, i)
				features.update(right_context[0])
				window = left_context[1]
				window.append(word['form'])
				window.extend(right_context[1])
			ngrams = self.ngrams(window)
			features.update(ngrams[0])
			features.update(ngrams[1])
		else:
			features['BOS'] = True
		return features

	def sent2features(self, sent):
		'''
		Все признаки для одного предложения.
		'''
		return [self.word2features(sent, i) for i in range(len(sent))]

	def sent2labels(self, sent):
		'''
		Все классы для одного предложения.
		'''
		return [sent[i]['label'] for i in range(len(sent))]

class Classifier():
	'''
	Классификаторы, обучение и прогнозирование.
	'''

	def training(self, X_train, y_train):
		self.crf = sklearn_crfsuite.CRF()
		self.crf.fit(X_train, y_train)

	# def predict(self, X_test):
	# 	y_pred = self.crf.predict(X_test)
	# 	return y_pred

	def pickle_model(self, name):
		'''
		Pickle модели.
		'''
		with open(name, 'wb') as f:
			pickle.dump(self.crf, f)

	def cross_validation(self, X_train, y_train):
		'''
		Кросс-валидация.
		:param X_train: 
		:param y_train: 
		:return: 
		'''
		scores = cross_val_score(self.crf, X_train, y_train, cv=5)
		print('Cross_validation_scores:', scores)

if __name__ == '__main__':
	feat_extr = FeatureExtr()
	result = feat_extr.load_conllu('chunking_data_test.txt')
	new_result = feat_extr.form_X(result)
	print('X_train processing starts...')
	X_train = [feat_extr.sent2features(sent) for sent in new_result]  # множество признаков
	pp.pprint(X_train)
	print('X_train is formed.')
	print('y_train processing starts...')
	y_train = [feat_extr.sent2labels(sent) for sent in new_result]  # классы
	print('y_train is formed.')
	#X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 0.2, random_state = 0)
	clfr = Classifier()
	print('Training starts...')
	clfr.training(X_train, y_train)  		# обучение модели
	clfr.cross_validation(X_train, y_train)


