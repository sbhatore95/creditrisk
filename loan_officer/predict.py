import abc
from .models import SavedState
from loan_admin.models import UploadFile, Criteria, CriteriaHelper, Configuration
from .project import *

class RuleBasedStrategyAbstract(object):
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def generate_score(self, loan_id):
		"""Required Method"""

class RuleBasedStrategy(RuleBasedStrategyAbstract):
	def generate_score(self, loan_id):
		f = open('test_id_dataset.csv', 'r')
		line = f.readline()
		array = line.split(',')
		array[-1] = array[-1].replace('\n', '')
		Dict = {}
		for i in range(0, len(array)):
			Dict[array[i]] = i
		line = f.readline()
		while line != "":
			sp = line.split(',')
			if(sp[0] == loan_id):
				break
			line = f.readline()
		sp[-1] = sp[-1].replace('\n', '')
		cri = Criteria.objects.all()
		conf = Configuration.objects.all()
		ans = 0
		for configuration in conf:
			for criteria in cri:
				helper = CriteriaHelper.objects.all().filter(criteria=criteria)
				truth = (criteria.feature == configuration.feature and 
					criteria.product == configuration.product and 
					criteria.category == configuration.category)
				if(truth):
					if(criteria.data_source == '3'): # for SQL
						feature = criteria.api.split(' ')[-1]
					if(feature in array):
						feature_score = 0
						for crhelper in helper:
							# For categorical
							tup = crhelper.entry.split(' ')
							if(tup[0] == "is"):
								if(tup[1] == sp[Dict[feature]]):
									feature_score += crhelper.score
							elif(tup[0] == ">"):
								if(int(sp[Dict[feature]]) > int(tup[1])):
									feature_score += crhelper.score
							elif(tup[0] == "<"):
								if(int(sp[Dict[feature]]) < int(tup[1])):
									feature_score += crhelper.score
							elif(tup[0] == ">="):
								if(int(sp[Dict[feature]]) >= int(tup[1])):
									feature_score += crhelper.score
							elif(tup[0] == "<="):
								if(int(sp[Dict[feature]]) <= int(tup[1])):
									feature_score += crhelper.score
						ans += configuration.weightage * feature_score
		f.close()
		return str(ans)

class DataBasedStrategyAbstract(object):
	__metaclass__ = abc.ABCMeta

	def parse(self, loan_id):
		f = open('test_id_dataset.csv', 'r')
		line = f.readline()
		sp = line.split(',')
		count = 0
		while(line != ""):
			if(sp[0] == loan_id):
				break
			sp = line.split(',')
			line = f.readline()
		del sp[0]
		del sp[0]
		return sp

	@abc.abstractmethod
	def predict(self, loan_id):
		"""Required Method"""

class StatisticalStrategy(DataBasedStrategyAbstract):
	def predict(self, loan_id):
		return load_and_predict("statistical", self.parse(loan_id))

class MLStrategy(DataBasedStrategyAbstract):
	def predict(self, loan_id):
		return load_and_predict("ml", self.parse(loan_id))

rs = RuleBasedStrategy()
ss = StatisticalStrategy()
ms = MLStrategy()

class ClassificationStrategy(object):
	def __init__(self, rule_strategy, data_strategy):
		self._rule_strategy = rule_strategy
		self._data_strategy = data_strategy

	def generate_score(self, loan_id):
		return self._rule_strategy.generate_score(loan_id)

	def predict(self, loan_id):
		return self._data_strategy.predict(loan_id)

class RuleBasedClassifier(ClassificationStrategy):
	def __init__(self):
		super(RuleBasedClassifier, self).__init__(rs, None)

class StatisticalBasedClassifier(ClassificationStrategy):
	def __init__(self):
		super(StatisticalBasedClassifier, self).__init__(None, ss)

class MLBasedClassifier(ClassificationStrategy):
	def __init__(self):
		super(MLBasedClassifier, self).__init__(None, ms)

class Classifier:
	def __init__(self, rule, stat, ml):
		self.rule = rule
		self.stat = stat
		self.ml = ml

	def doClassification(self, loan_id):
		ans = [None, None, None]
		if(self.rule):			
			classifier = RuleBasedClassifier()
			ans[0] = classifier.generate_score(loan_id)
		if(self.stat):
			classifier = StatisticalBasedClassifier()
			ans[1] = classifier.predict(loan_id)
		if(self.ml):
			classifier = MLBasedClassifier()
			ans[2] = classifier.predict(loan_id)
		return ans