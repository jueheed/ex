'''
Created on April 01, 2020

@author: hong
'''

import unittest
# This is the class we want to test. So, we need to import it
from train_ner_example import TrainNerEx as TrainNerExClass

#common features for other three classes
class CommonUnitTest(unittest.TestCase):
    @staticmethod
    def get_text(entity):
        return entity[0]
    @staticmethod    
    def get_type(entity, entity_type):
        return entity[1] == entity_type

#load first short message in which a train data has been created against what expected, catch entities, then test
class TrainNerData1Test(CommonUnitTest):
    expected_org_in_data1=['FinTRAC','Capital One Bank USA','Capital One', 'Capital One Bank','Walmart','Target','Quality Inn','Best Buy' ]
    expected_date_in_data1=['1/12/2017','3/15/2016','5/14/2016','5/12/2016','5/14/2016']
    expected_gpe_in_data1=['NA','Canada']
    expected_money_in_data1=['$8,098.76','$8,098.76','$30.03']
    
    @classmethod
    def setUpClass(cls):
        #fetch entities
        cls.entities = TrainNerExClass().ner_search('data/data1.txt')

    @classmethod
    def tearDownClass(cls):
        #clear
        cls.entities=[]
        

    def test_ner_search_by_org(self):

        actual_org_list=[ str(super(TrainNerData1Test, self).get_text(entity)) for entity in self.entities if super(TrainNerData1Test, self).get_type(entity,'ORG') ]
        
        lst=[value for value in self.expected_org_in_data1 if value in actual_org_list]
        actual_percentage=len(lst)/len(self.expected_org_in_data1)
        self.assertEqual(actual_percentage,100/100)
        
    def test_ner_search_by_gpe(self):
        actual_gpe_list=[str(super(TrainNerData1Test, self).get_text(entity)) for entity in self.entities if super(TrainNerData1Test, self).get_type(entity,'GPE') ]
            
        lst=[value for value in self.expected_gpe_in_data1 if value in actual_gpe_list]
        actual_percentage=len(lst)/len(self.expected_gpe_in_data1)
        self.assertEqual(actual_percentage,100/100)

    def test_ner_search_by_date(self):
        actual_date_list=[str(super(TrainNerData1Test, self).get_text(entity)) for entity in self.entities if super(TrainNerData1Test, self).get_type(entity,'DATE') ]
            
        lst=[value for value in self.expected_date_in_data1 if value in actual_date_list]
        actual_percentage=len(lst)/len(self.expected_date_in_data1)
        self.assertEqual(actual_percentage,100/100)
        
    def test_ner_search_by_money(self):
        actual_money_list=[str(super(TrainNerData1Test, self).get_text(entity)) for entity in self.entities if super(TrainNerData1Test, self).get_type(entity,'MONEY') ]
            
        lst=[value for value in self.expected_money_in_data1 if value in actual_money_list]
        actual_percentage=len(lst)/len(self.expected_money_in_data1)
        self.assertEqual(actual_percentage,100/100)
        
        
if __name__ == '__main__':
    unittest.main()
