import pandas as pd
import numpy as np
import lightgbm as lgb 
import streamlit as st
import datetime

@st.cache()

def store_sales_forecast(date,store,item,id):

  X = [[date,store,item,id]]
  test_data = pd.DataFrame(np.array(X),columns=['date','store','item','id'])
  test_data['date'] = pd.to_datetime(test_data['date'])
  test_data = test_data.astype({"store": int, "item": int, "id": float})
  test_data['day'] = test_data.date.dt.day
  test_data['month'] = test_data.date.dt.month
  test_data['year'] = test_data.date.dt.year
  test_data['dayofweek'] = test_data.date.dt.dayofweek
  store_2018 = pd.read_csv('store_2018.csv',parse_dates=['date'])
  store = test_data
  store['sales'] = store_2018.loc[(store_2018['date'] == store.loc[0,'date']) & (store_2018['store'] == store.loc[0,'store']) & (store_2018['item'] == store.loc[0,'item'])]['sales']
  
  store['dayofyear'] = store.date.dt.dayofyear
  store['weekofyear'] = store.date.dt.weekofyear
  store['weekend_yes'] = store.date.dt.weekday // 4
  store['month_start_yes'] = store.date.dt.is_month_start.astype(int)
  store['month_end_yes'] = store.date.dt.is_month_end.astype(int)
  store['quarter'] = store.date.dt.quarter
  store['weekofmonth'] = store['weekofyear'].values // 4.35                                                                                                                                                                               
  store['mon_yes'] = np.where(store['dayofweek'] == 0, 1, 0)                                                                                            
  store['tue_yes'] = np.where(store['dayofweek'] == 1, 1, 0)                                                                                         
  store['wed_yes'] = np.where(store['dayofweek'] == 2, 1, 0)                                                                                         
  store['thu_yes'] = np.where(store['dayofweek'] == 3, 1, 0)                                                                                         
  store['fri_yes'] = np.where(store['dayofweek'] == 4, 1, 0)                                                                                         
  store['sat_yes'] = np.where(store['dayofweek'] == 5, 1, 0)                                                                                         
  store['sun_yes'] = np.where(store['dayofweek'] == 6, 1, 0) 

  exp_time_features = ['dayofweek', 'weekofmonth', 'weekofyear', 'month', 'quarter', 'weekend_yes'] 
  for exp_item in exp_time_features:
    expanding_store = store.groupby(['store', 'item', exp_item])['sales'].expanding().mean().bfill().reset_index()
    expanding_store.columns = ['store', 'item', exp_item, 'exp_index', 'exp_'+exp_item]
    expanding_store = expanding_store.sort_values(by=['item', 'store', 'exp_index'])
    store['exp_'+exp_item] = expanding_store['exp_'+exp_item].values

  store.sort_values(by=['item', 'store', 'date'], axis=0, inplace=True)

  #Adding Lag values as feature
  l = [8,15,22,29,30,31,38,61,67,73,91, 98, 105, 112, 180, 270, 365, 546, 728]                                                                                                                                                                                                                      
  for var_l in l:                                                                                                                          
    store['l_' + str(var_l)] = store.groupby(["item", "store"])['sales'].transform(lambda y: y.shift(var_l)) + np.random.normal(scale=0.01, size=(len(store),))  

  #Adding Rolling Mean values as feature
  r = [8,15,22,29,30,31,38,61,67,73,91, 98, 105, 112, 180, 270, 365, 546, 728]                                                                                                                                                                                                                                                                                                                       
  for var_r in r:                                                                                                                    
    store['r_' + str(var_r)] = store.groupby(["item", "store"])['sales'].transform(lambda y: y.shift(1).rolling(window=var_r, min_periods=8, win_type="triang").mean()) + np.random.normal(scale=0.01, size=(len(store),)) 

  #Adding Exponentially Mean values as feature
  ewm_a = [0.95, 0.9, 0.8, 0.7, 0.5,.4,.3,.2,.1]                                             
  ewm_l = [8,15,22,29,30,31,38,61,67,73,91, 98, 105, 112, 180, 270, 365, 546, 728]                                                                                                      
  for var_a in ewm_a:                                                                                                                      
    for var_l in ewm_l:                                                                                                                      
      store['ewm_a_' + str(var_a) + "_l_" + str(var_l)] = store.groupby(["item", "store"])['sales'].transform(lambda y: y.shift(var_l).ewm(alpha=var_a).mean()) 

  store_encoding = pd.get_dummies(store[['store', 'item', 'dayofweek', 'month']], columns=['store', 'item', 'dayofweek', 'month'], dummy_na=True)  
  store_final = pd.concat([store, store_encoding], axis=1)                                                                                                          
  store_lgbm_columns = [column for column in store_final.columns if column not in ['date', 'id', 'sales', 'year']]                                                                                                            
  test = store_final[store_lgbm_columns] 
                                                                                                                                                                                                                                                                                                                   
  model = lgb.Booster(model_file='store_lgbm_model.txt')
  store_lgbm_preds = model.predict(test, num_iteration=1500) 
  store_lgbm_preds_sales = np.round(np.expm1(store_lgbm_preds),0)

  return int(store_lgbm_preds_sales[0])

def main():       

    case_study_store_box = """ 
    <div style ="background-color:orange;padding:5px"> 
    <h1 style ="color:black;text-align:center;">Project 1</h1> 
    <h1 style ="color:black;text-align:center;">Store Item Demand Forecasting</h1> 
    </div> 
    """
    st.markdown(case_study_store_box, unsafe_allow_html = True)

    store_date = st.date_input("Date", value=datetime.date(2018, 1, 1), min_value=datetime.date(2018, 1, 1), max_value=datetime.date(2018, 3, 31), help="Please enter date for which sales will be forecasted")
    
    store_number = st.number_input("Store", min_value=1, max_value=10, value=1, help="Please enter Store Id")
    
    store_item = st.number_input("Item", min_value=1, max_value=50, value=1, help="Please enter Item Id")

    if st.button("Sales"): 
        sales_value = store_sales_forecast(store_date,store_number,store_item,0) 
        st.success('{}'.format(sales_value))
     
if __name__=='__main__': 
    main()