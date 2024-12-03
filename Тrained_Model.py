#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st


# In[106]:


from sklearn import linear_model
import scipy.stats as stat

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):#,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)

    def fit(self,X,y):
        self.model.fit(X,y)
        
        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values
        
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values




with open('model_test_pkl' , 'rb') as f:
    lr = pickle.load(f)




car_dict = {"vehicle_mark:Bottom":['ACURA','ZNEN','LINCOLN','CAGIVA','MALAGUTI','ISUZU','HUMMER','PORSCHE','CADILLAC',
                                'DODGE','CHRYSLER','IVECO'], 
            'vehicle_mark:BMW':['BMW'],
          "vehicle_mark:SEAT_AUDI" :['SEAT','AUDI'],
            "vehicle_mark:MERCEDES-BENZ": ['MERCEDES-BENZ'],
          'vehicle_mark:MI_OPEL_HONDA':['MITSUBISHI','OPEL','HONDA'],
         'vehicle_mark:FORD_TO_RENAULT':['FORD','ALFA ROMEO','VOLKSWAGEN','PEUGEOT','NISSAN','SUBARU','INFINITI','DACIA',
                                         'MAZDA','CHEVROLET','RENAULT'],
        'vehicle_mark:VOLVO_TO_TOP':['VOLVO','LANCIA','KIA','CITROËN','FIAT','JAGUAR','SUZUKI','MINI','LAND ROVER','TOYOTA',
                                 'ŠKODA','HYUNDAI','JEEP','SSANGYONG','YAMAHA','WIOLA','DAEWOO','DAIHATSU','DR MOTOR',
                                 'GREAT WALL','MAN', 'SMART','SHENYANG','SAAB','ROVER','KAWAZAKI','LEXUS','PIAGGIO',
                                'LEXMOTO','OLDSMOBILE'],
         }

city_dict = {'District:GAB_TO_MON':['ГАБРОВО','ВИДИН','ЛОВЕЧ','ПЛЕВЕН','ВРАЦА','РУСЕ','МОНТАНА'],
            'District:HAS_TO_BUR': ['ХАСКОВО','ДОБРИЧ','БУРГАС'],
            'District:TAR_TO_KYST': ['ВЕЛИКО ТЪРНОВО','ПЕРНИК','КЮСТЕНДИЛ'],
             "District:BLG_TO_SIL":['БЛАГОЕВГРАД','ТЪРГОВИЩЕ','ЯМБОЛ','СОФИЙСКА ОБЛАСТ','СИЛИСТРА'],
             'District:ОБЛАСТ ВАРНА':['ВАРНА'],
             "District:ОБЛАСТ СОФИЯ - ГРАД":['СОФИЯ-ГРАД'],
             'District:ОБЛАСТ СТАРА ЗАГОРА':['СТАРА ЗАГОРА'],
            'District:PAZ_TO_PLV':['ПАЗАРДЖИК','ПЛОВДИВ'],
            'District:SML_TO_SHUM':['СМОЛЯН','КЪРДЖАЛИ','СЛИВЕН','РАЗГРАД','ШУМЕН']}

service_dict = {"service_type:LEASING":'LEASING',
               'service_type:LEASEBACK':'LEASEBACK'}

gender_dict = {"Gender:M":'M',
               'Gender:F':'F'}




columns_dict = {'Gender:F':0,'Gender:M':0,
'service_type:LEASEBACK':0,
'service_type:LEASING':0,
'vehicle_mark:Bottom':0,
'vehicle_mark:BMW':0,
'vehicle_mark:SEAT_AUDI':0,
'vehicle_mark:MERCEDES-BENZ':0,
'vehicle_mark:MI_OPEL_HONDA':0,
'vehicle_mark:FORD_TO_RENAULT':0,
'vehicle_mark:VOLVO_TO_TOP':0,
'District:GAB_TO_MON':0,
'District:HAS_TO_BUR':0,
'District:TAR_TO_KYST':0,
'District:ОБЛАСТ ВАРНА':0,
'District:ОБЛАСТ СТАРА ЗАГОРА':0,
'District:BLG_TO_SIL':0,
"District:ОБЛАСТ СОФИЯ - ГРАД":0,
'District:PAZ_TO_PLV':0,
'District:SML_TO_SHUM':0,
'term_value_bin:0-17.7':0,
'term_value_bin:17.7-25.5':0,
'term_value_bin:25.5-45':0,
'term_value_bin:45-64.5':0,
'term_value_bin:64.5-90':0,
'ltv_actual_bin:0-0.6':0,
'ltv_actual_bin:>0.6':0,
'age_at_issue_bin:17-28.6':0,
'age_at_issue_bin:28.6-42.733':0,
'age_at_issue_bin:>42.733':0}



def term_funct(n):
    if n <= 17.7:
        term_value = 'term_value_bin:0-17.7'
    elif n > 17.7 and n <= 25.5:
        term_value = 'term_value_bin:17.7-25.5'
    elif n > 25.5 and n <= 45.0:
        term_value = 'term_value_bin:25.5-45'
    elif n > 45.0 and n <= 64.5:
        term_value = 'term_value_bin:45-64.5'
    elif n > 64.5 and n <= 90.0:
        term_value = 'term_value_bin:64.5-90'
        
    return term_value




def ltv_funct(n):
    if n <= 0.6:
        ltv_value = 'ltv_actual_bin:0-0.6'
    elif n > 0.6:
        ltv_value = 'ltv_actual_bin:>0.6'
    return ltv_value



def age_funct(n):
    if n => 17 and n <= 28.6:
        age_value = 'age_at_issue_bin:17-28.6'
    if n > 28.6 and n <= 42.733:
        age_value = 'age_at_issue_bin:28.6-42.733'
    if n > 42.733 and n <= 120:
        age_value = 'age_at_issue_bin:>42.733'
    return age_value




town_list = city_dict.values()
town_tuple = tuple(tuple(i) for x,i in enumerate(town_list))
fin_town_tuple = tuple(j for i in town_tuple for j in i)

car_list = car_dict.values()
car_tuple = tuple(tuple(i) for x,i in enumerate(car_list))
fin_car_tuple = tuple(j for i in car_tuple for j in i)

type_list = service_dict.values()
fin_type_tuple = tuple(type_list)

gender_list = gender_dict.values()
fin_gender_tuple = tuple(gender_list)


with st.sidebar:
    st.header('Input features')
    town = st.selectbox('Town', fin_town_tuple)
    vehicle = st.selectbox('Vehicle', fin_car_tuple)
    service = st.selectbox('Service Type', fin_type_tuple)
    gender = st.selectbox('Gender', fin_gender_tuple)
    
    term_value = st.slider('Term', 0, 90, 50)
    ltv_value = st.slider('LTV', 0.0, 1.0, 0.5)
    age_value = st.slider('Age', 17, 120, 50)

#st.title('Credit Risk Model')
   
#col = st.columns((3,3), gap='medium')

town_index = [k for k, v in city_dict.items() if town in v][0]
vehicle_index = [k for k, v in car_dict.items() if vehicle in v][0]
age_index = age_funct(float(age_value))
ltv_index = ltv_funct(float(ltv_value))
service_index = [k for k, v in service_dict.items() if service in v][0]
gender_index = [k for k, v in gender_dict.items() if gender in v][0]
term_index = term_funct(float(term_value))



var_list = [town_index, vehicle_index, age_index, ltv_index, service_index, gender_index, term_index]


empty_frame = pd.DataFrame(columns_dict, index = [0])


empty_frame[var_list] = 1



ref_categories = ['Gender:M',
'service_type:LEASEBACK',
'vehicle_mark:Bottom',
'District:GAB_TO_MON',
'term_value_bin:64.5-90',
'ltv_actual_bin:>0.6',
'age_at_issue_bin:17-28.6']


inputs_test = empty_frame.drop(ref_categories, axis=1)
#results = lr.model.predict(inputs_test)
threshold = 0.5




prediction = lr.model.predict(inputs_test)
prediction_proba = lr.model.predict_proba(inputs_test)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['TERMINATED', 'PAID']
df_prediction_proba.rename(columns={0: 'TERMINATED',
                                 1: 'PAID'})
df_prediction_proba = round(df_prediction_proba,2)                 
st.subheader('Predicted Status')

st.dataframe(df_prediction_proba,
             column_config={
               'TERMINATED': st.column_config.ProgressColumn(
                 'TERMINATED',
                 format='%f',
                 width='medium',
                 min_value=0.0,
                 max_value=1.0
               ),
               'PAID': st.column_config.ProgressColumn(
                 'PAID',
                 format='%f',
                 width='medium',
                 min_value=0.0,
                 max_value=1.0
               ), }, hide_index=True)
               
statuses = np.array(['Terminated', 'Paid'])
st.success(str(statuses[prediction][0]))










