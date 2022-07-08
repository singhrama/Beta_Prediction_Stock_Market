# -*- coding: utf-8 -*-
#!/usr/bin/env python3


import sys

'''
sys.path.insert(0,"/geode2/home/u080/singrama/Carbonate/.local/lib/python3.6/site-packages")
'''

import pandas as pd
import numpy as np
from glob import glob
import os
import wrds
import pickle
import time
import json
from sklearn.linear_model import LinearRegression
import option_util as util
import datetime
import shutil
import statistics
import pickle5 as p
import glob
import math
import warnings
import logging
warnings.filterwarnings('ignore')
    
def setup_logging(inti_file, chunkid):

    ###Gets or creates a logger
	logger = logging.getLogger(__name__)  
    
    # set log level
	logger.setLevel(logging.INFO)

    # define file handler and set formatter
    #file_handler = logging.FileHandler('logfile.log')
	out_files = os.path.join(inti_file, 'log')
	if not os.path.isdir(out_files): 
		os.mkdir(out_files)
	logfilepath = out_files + '/'+ 'logfile_chunkid_'+ str(chunkid) + "_" +str(datetime.datetime.now()).replace(":","_") + '.log'
        
	file_handler = logging.FileHandler(logfilepath)
	formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
	file_handler.setFormatter(formatter)

    # add file handler to logger
	logger.addHandler(file_handler)
    
	return logger
    
def cond_bea(onew_sec_cusip, inti_file,cus,oyr,ologger,vsurf_path,secprd_path):

    ### get all optionm (option metrics) tables
#     tables = conn.list_tables(library = 'optionm')
#     tables = sorted(tables)

#     ### read in security info
#     secinfo = conn.get_table(library = 'optionm', table = 'securd')
#     secinfo = secinfo.loc[secinfo.cusip == str(cus), :]

# #     print(secinfo)
#     secid = int(secinfo.secid.values[0])
    
	secid = onew_sec_cusip.get(cus)
	cusip_id = cus
	print(cus, secid, oyr)
	files = glob.glob(vsurf_path + '/'+  oyr + '*')
    
	data = pd.DataFrame()
	for i in files:
		with open(i, 'rb') as vsuf_pkl:
			odata = pickle.load(vsuf_pkl)
			data  = pd.concat([data, odata[odata.secid.isin([secid])]])
        
    ### read in SPDR data
#     data = conn.raw_sql("""select * 
#                     from optionm.vsurfd""" + str(oyr)+
#                     """ where secid = """ + str(secid))
    
	if not(data.empty): 
#     data = conn.raw_sql("""select * 
#                         from optionm.vsurfd2014 
#                         where secid = """ + str(secid))
#     print(data)
    ### merge to security info
    
#         data = data.merge(secinfo, on = 'secid')

        ### get prices
        #secprc = conn.get_table(library = 'optionm', table = 'secprd2014')
        
		files = glob.glob(secprd_path + '/'+  oyr + '*')

		for i in files:
			with open(i, 'rb') as secprd_pkl:
				secprc = pickle.load(secprd_pkl)
				secprc  = secprc[secprc.secid.isin(['secid'])]
        
#         secprc = conn.raw_sql("""select secid, date, close
#                               from optionm.secprd""" + str(oyr)+
#                               """ where secid = """ + str(secid))

        #secprc = conn.raw_sql("""select secid, date, close
                              #from optionm.secprd2014
                              #where secid = """ + str(secid))

		secprc.rename(columns = {'close': 'spot'}, inplace = True)
		data = data.merge(secprc, on = ['secid', 'date'])

        ### save data
        #data.to_pickle(datadir + 'aapl_surface_2014.pkl')
		data['date']=pd.to_datetime(data['date'])
		data['date'] = data['date'].dt.strftime("%Y%m%d")
        
		if not(data.empty):
			data = data.dropna(axis=0, subset=['impl_volatility']) #deleting rows where implied volatility is null  
			df_rf=pd.read_csv(inti_file + '/' + 'RF.csv',index_col=[0])
			df_rf=df_rf[0:1145]
			df_rf.drop(labels=['MktRF', 'smb','hml'], axis=1,inplace = True)
			df_rf['rf']=(df_rf['rf']*12)/100
			df_rf['yandm'] = pd.to_datetime(df_rf['yandm'], format='%Y%m')
			df_rf['yandm'] = df_rf['yandm'].dt.strftime("%Y%m")
			df_rf[df_rf['yandm']==data['date'].unique()[0][:-2]].iloc[0,1]

			dataframe_new=pd.DataFrame(columns = ['date','ttm(in yrs)','P/S'])

			for i in range(len(data['date'].unique())):
				dated=data['date'].unique()[i]
				risk_free_rate=df_rf[df_rf['yandm']==data['date'].unique()[i][:-2]].iloc[0,1]
				df=data[data['date']==dated]
				for j in range(len(df['days'].unique())):
					time_to_maturity=df['days'].unique()[j]
					time_to_maturity_yrs=(time_to_maturity/365.0)  #Converting ttm to years from months
					df_new=data[(data['date']==dated) & (data['days']==time_to_maturity)]
				values=[]
				df_final_C=data[(data['date']==dated) & (data['days']==time_to_maturity) & (data['cp_flag']=='C')]   #Call option
				df_final_P=data[(data['date']==dated) & (data['days']==time_to_maturity) & (data['cp_flag']=='P')]   #Put Option
				length=len(df_final_C)
				my_svix2 = util.svix2(np.flip(df_final_P.impl_strike.values), np.flip(df_final_P.impl_premium.values), np.flip(df_final_C.impl_strike.values), np.flip(df_final_C.impl_premium.values), np.mean(df_new.spot))
				my_svix2.clean()

				if my_svix2.check() == True:
					my_svix2.fit()

					for k in range(length):
						strikeprice_call=df_final_C.iloc[k,5]  
						spot_price=df_final_C.iloc[k,17]
						call_option_price=df_final_C.iloc[k,6]  #call_option_premium
						put_option_price = my_svix2.put_func(strikeprice_call)
						exp=((put_option_price-call_option_price+spot_price-strikeprice_call*math.exp(risk_free_rate*time_to_maturity_yrs))/spot_price)
						values.append(exp)
					for l in range(length):
						strikeprice_put=df_final_P.iloc[k,5]  
						spot_price=df_final_P.iloc[k,17]
						put_option_price=df_final_P.iloc[k,6]  #put_option_premium
						call_option_price = my_svix2.call_func(strikeprice_put)
						exp=((put_option_price-call_option_price+spot_price-strikeprice_put*math.exp(risk_free_rate*time_to_maturity_yrs))/spot_price)
						values.append(exp)

                #finding median across strike prices
					if len(values)!=0:
						med_value=statistics.median(map(float,values))
						dataframe_new.loc[len(dataframe_new.index)]=[dated,time_to_maturity_yrs,med_value]  #adding the record to our new dataframe

				else:
					ologger.warning(cusip_id + '_Incomplete_Data_SVIX_Check_For_' + oyr)

				dataframe_new['P/S']=dataframe_new['P/S']*(-1)
				dataframe_new['date']=pd.to_datetime(dataframe_new['date'],format="%Y%m%d")
				dataframe_new['date'] = dataframe_new['date'].dt.strftime("%Y%m%d")
				dataframe_nn_2=pd.DataFrame(columns = ['date','div_comp1_slope','div_comp1_intercept'])
				for i in range(len(dataframe_new['date'].unique())):
					d=dataframe_new['date'].unique()[i]
					df_calculate_2=dataframe_new[dataframe_new['date']==d]
					m, b = np.polyfit(df_calculate_2['ttm(in yrs)'], df_calculate_2['P/S'], 1)
					dataframe_nn_2.loc[len(dataframe_nn_2)]=[d,m,b]  #adding the record to our new dataframe
                #print(dataframe_nn_2)

			dir_pickle = "pickel_files_" + cusip_id 
			outdir = os.path.join(inti_file, dir_pickle)
			if os.path.isdir(outdir) ==False:  os.mkdir(outdir)

            #print(outdir)

            ### index by date * ttm 
			N = data[['date', 'days']].drop_duplicates().shape[0]
			bar = util.bar(N)
			kk = 0
			data.index = data.date

			undates = np.unique(data.date)

			for date in undates:
				date_data = data.loc[date, :]
				ttms = np.unique(date_data.days)
				for ttm in ttms:
					sdata = date_data.loc[date_data.days == ttm, :]
					sdata.index = np.arange(sdata.shape[0])
					savefile = outdir + '/' + str(date)[:10] + '_' + str(ttm) + '.pkl.bz2'
					sdata.to_pickle(savefile)
					kk += 1
					bar.update(kk)

            ### finish
			bar.finish()

            ### read in chunks 
			files = sorted(os.listdir(outdir))
            #print(files)
			N = len(files)
			bar = util.bar(N)
			opt_data = []
			for ii, file in enumerate(files):

				bar.update(ii)

				data1 = pd.read_pickle(outdir + '/' + file)
				puts = data1.loc[data1.cp_flag == 'P', :]
				calls = data1.loc[data1.cp_flag == 'C', :]

				om = util.svix2(np.flip(puts.impl_strike.values), 
                                np.flip(puts.impl_premium.values), 
                                np.flip(calls.impl_strike.values), 
                                np.flip(calls.impl_premium.values),
                                np.mean(data1.spot))

				success = om.calc_svix2()
				if success == False:
					continue
				trf_svix2 = om.trf_svix2
				forward = om.forward
				ttm = np.mean(data1.days)
				date = data1.date[0]
                #exdate = data.exdate[0]
				impl_vol=data1.impl_volatility[0]
                #bid_ask=data.bid_ask[0]
				strike=data1.impl_strike[0]

                ### compile data
				opt_data.append({'trf_svix2': trf_svix2, 'forward': forward,
                                 'ttm': ttm, 'date': date,  'impl_vol':impl_vol,'strike':strike})

            ### finish and save
			bar.finish()
			opt_data = pd.DataFrame(opt_data)

            #print('finished')
            #print(opt_data)
			dataframe_nn_3=pd.DataFrame(columns = ['date','svix_slope','svix_intercept'])

			for i in range(len(opt_data['date'].unique())):
				d=opt_data['date'].unique()[i]
				df_calculate_3=opt_data[opt_data['date']==d]
				m, b = np.polyfit(df_calculate_3['ttm'], df_calculate_3['trf_svix2'], 1)
				dataframe_nn_3.loc[len(dataframe_nn_3)]=[d,m,b]  #adding the record to our new dataframe

			stocksdata=data
			stocksdata['date']=pd.to_datetime(stocksdata['date'])
			stocksdata['ss1']=stocksdata['impl_strike']/stocksdata['spot']
			stocksdata['ss2']=stocksdata['ss1']**2
			stocksdata['ss3']=stocksdata['ss1']**3
			stocksdata['ss1t']=stocksdata['ss1']*stocksdata['days']
			stocksdata['ss2t']=stocksdata['ss2']*stocksdata['days']
			stocksdata['ss3t']=stocksdata['ss3']*stocksdata['days']

			dataframe_nn_1_initial=pd.DataFrame(columns = ['date','cp_flag','Coefficient_1','Coefficient_2','Coefficient_3','Coefficient_4','Coefficient_5','Coefficient_6','Coefficient_7','Intercept'])
			for i in range(len(stocksdata['date'].unique())):
				d=stocksdata['date'].unique()[i]
				for j in range(len(stocksdata['cp_flag'].unique())):
					option_type=stocksdata['cp_flag'].unique()[j]
					dataframe_req=stocksdata[(stocksdata['date']==d) & (stocksdata['cp_flag']==option_type)]
					model = LinearRegression()
					y=dataframe_req['impl_volatility']
					x=dataframe_req[['date','cp_flag','impl_strike','ss1','ss2','ss3','ss1t','ss2t','ss3t','days','impl_volatility']]
					model.fit(x[x.columns[3:-1]], y)
					y_pred = model.predict(x[x.columns[3:-1]])
					x['pred_volatility']=y_pred.tolist()
					dataframe_nn_1_initial.loc[len(dataframe_nn_1_initial.index)]=[d,option_type,model.coef_[0],model.coef_[1],model.coef_[2],model.coef_[3],model.coef_[4],model.coef_[5],model.coef_[6],model.intercept_]  #adding the record to our new dataframe

			dataframe_nn_1_initial['date']=pd.to_datetime(dataframe_nn_1_initial['date'])
			dataframe_nn_1_initial['date'] = dataframe_nn_1_initial['date'].dt.strftime("%Y%m%d")
            #print(dataframe_nn_1_initial)
			df_P=dataframe_nn_1_initial[dataframe_nn_1_initial['cp_flag']=='P']
			df_C=dataframe_nn_1_initial[dataframe_nn_1_initial['cp_flag']=='C']
			m1=df_P.merge(df_C, on=["date"])
			m2=m1.merge(dataframe_nn_3, on=["date"])
			df_final=m2.merge(dataframe_nn_2, on=["date"])

			df_final.insert(loc=0, column='cusip_id', value=cusip_id)

			df_final.drop(['cp_flag_x','cp_flag_y'], axis = 1, inplace=True)
			out_files = os.path.join(inti_file, cus)
  
			if os.path.isdir(out_files) ==False:  os.mkdir(out_files)
			df_final.to_csv(out_files + '//' +'Final_File_' + cus + '_'+ oyr + '.csv', index= False)
			ologger.info(cus + '_Completed_For_' + oyr)
			if os.path.isdir(outdir) ==True: shutil.rmtree(outdir, ignore_errors=True)

		else:
			ologger.warning(cusip_id + '_Implied_Volatility_Is_Null_' + oyr)
    
	else:
		ologger.warning(cusip_id + '_No_Data_For_' + oyr)

def check_cusip_exist(inti_path,ocusip,ologger):
	otrack = inti_path + '/' + "CUSIP_TRACKER.json"
	if os.path.exists(otrack):
		with open(otrack, 'r') as jf:
#             print(otrack)
			data = read_json(jf)

		if not (data.get(ocusip)):
			return True    # If new cusip 
		else:
			ologger.info(ocusip + '_Already_Downloaded')
			return False   # If cusip already downloaded
	else:
		return True  # If new cusip

def read_json(jf):
	i=0
	for i in range(5):
		try:
			data = json.load(jf)
			return data
		except ValueError:
			time.sleep(10)
			i+=1

def add_cusip_exist(inti_path,ocusip,ologger):
	otrack = inti_path + '/' + "CUSIP_TRACKER.json"
	if os.path.exists(otrack):
		with open(otrack, 'r') as jf:
			data = read_json(jf)
#             print(otrack)
  
		if not (data.get(ocusip)):
			data.update({ocusip:ocusip})
			with open(otrack, 'w') as jf:
				json.dump(data, jf)
				jf.close()
				
#             return True    # If new cusip 
#         else:
#             ologger.info(ocusip + '_Already_Downloaded')
#             return False   # If cusip already downloaded
	else:
		with open(otrack, 'w') as jf:
			json.dump({ocusip:ocusip}, jf)
			jf.close()
#         return True  # If new cusip

def start_cusip():
	inti_file= "/N/u/singrama/Carbonate/Documents/Beta_Conditional/Final_inputs" #"D:\IUB\Course_Work\RA\Stock_Beta_Estimation\code"
	vsurf_path = '/N/project/conditional_betas/data/yearly_data/vsurfd/Final_Bz2_Files'
	secprd_path = '/N/project/conditional_betas/data/yearly_data/secprd/Final_Bz2_Files'
	json_files = '/N/project/conditional_betas/Chunk_Files/CUSIP_Pair_Run_JSON/'   
    
	os.chdir(inti_file)
    
	#opath = inti_file + '/vsurfpd_cusip_data.csv'
	#ocusips_vals_sec = pd.read_csv(opath)
    
	args = util.get_args()
	chunk_id = args['chunkid']
    #chunk_id = str(sys.argv).split('chunkid')[1][4:(len(str(sys.argv).split('chunkid')[1])-2)]
	print("Chunk_id : ",chunk_id)
	with open(json_files + "CUSIPs_" + str(chunk_id) + '.json', 'r') as jf:
		onew_sec_cusip = json.load(jf)

	onew_sec_cusip=dict(onew_sec_cusip)
	
	if not bool(onew_sec_cusip):
		print("CUSIP JSON File is Empty with Chunkid: ",chunk_id)
		return

	#onew_sec_cusip = dict(zip(ocusips_vals_sec.cusip, ocusips_vals_sec.secid))
    
	with open(inti_file + '/Years_Extract.txt') as f:
		lines = f.readlines()
		oyears = lines[0].split(",")

	ologger = setup_logging(inti_file, chunk_id)

	#for tick in tickers.ticker:
	for cusip in onew_sec_cusip:
		#print(tick)
		os.chdir(inti_file)
		
		if check_cusip_exist(inti_file,cusip,ologger):
			
			for year in oyears:
# 				try:

					ologger.info(cusip + '_Started_For_' + year)
                    
					cond_bea(onew_sec_cusip, inti_file, cusip, year.strip(), ologger, vsurf_path,secprd_path)
					ologger.info(cusip + '_File_Executed_For_' + year)

# 				except:
#					ologger.critical(cusip + '_For_' + year + '_FAILED')

		#         cond_bea(inti_file, 'MXIM', year.strip())
				#cond_bea(inti_file, tick, year)
				
			add_cusip_exist(inti_file,cusip,ologger)
			path = inti_file + '/' + cusip

			if os.path.isdir(path) ==True:
				all_filenames = [i for i in glob.glob(os.path.join(path, "*.csv"))]
			#combine all files in the list
				combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
				
				pcklepath = '/N/project/conditional_betas/Attribute_Data' #/N/project/conditional_betas/data/Final_Data' #'/N/slate/singrama/Final_Pickle_Files'
				if os.path.isdir(pcklepath)==False: os.mkdir(pcklepath)
					
			#export to csv
				pcklepath = pcklepath+ '/combined_pickle_' + cusip + '.p'
				pickle.dump( combined_csv, open( pcklepath, "wb" ) )

				if os.path.isdir(path) ==True: shutil.rmtree(path, ignore_errors=True)
				ologger.info(cusip + '_File_Done_With_Pickle_File')
			
def main():
	start_cusip()
	
if __name__=="__main__":
	#sys.path.insert(0,"/geode2/home/u080/singrama/Carbonate/.local/lib/python3.6/site-packages")
	main()	
