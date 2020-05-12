# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:16:18 2020

@author: Admin
"""

import json
from scipy.special import gamma
import math
import pandas as pd
import requests
from scipy import optimize
import math
import threading
import datetime
import numpy as np

with open("D:\\Covid-19\\data\\param.json", "r") as read_file:
    param = json.load(read_file)
    

def calculate_recs_gamma(n,a_inf,b_inf,k_inf,teta_inf,a_dead,b_dead,k_dead,teta_dead,a_inc,b_inc,k_inc,teta_inc):
    recs_pred = 0
    for j in range(n+1):
        inf_ch = 0
        if j>0:
            inf_ch = (max(0,(b_inf*((j-a_inf)**(k_inf-1)*
                              ((math.e**(-(j-a_inf)/teta_inf))/(teta_inf**k_inf*gamma(k_inf))))).real) - 
            max(0,(b_inf*((j-1-a_inf)**(k_inf-1)*
                              ((math.e**(-(j-1-a_inf)/teta_inf))/(teta_inf**k_inf*gamma(k_inf))))).real))
        else:
            inf_ch = max(0,(b_inf*((j-a_inf)**(k_inf-1)*
                              ((math.e**(-(j-a_inf)/teta_inf))/(teta_inf**k_inf*gamma(k_inf))))).real)
        recs_pred+= max(0,(max(0,(b_inc*((j-a_inc)**(k_inc-1)*
                              ((math.e**(-(j-a_inc)/teta_inc))/(teta_inc**k_inc*gamma(k_inc))))).real) - 
                     inf_ch - max(0,(b_dead*((j-a_dead)**(k_dead-1)*
                              ((math.e**(-(j-a_dead)/teta_dead))/(teta_dead**k_dead*
                                                                  gamma(k_dead))))).real)))
    return recs_pred

class Optimizer:
    def __init__(self, t,ks,data):
        self.k_if=param[t]['Original'][2]
        self.teta_if=param[t]['Original'][3]
        self.k_d=param[t]['Original'][6]
        self.teta_d=param[t]['Original'][7]
        self.k_ic=param[t]['Original'][10]
        self.teta_ic=param[t]['Original'][11]
        
        self.s = ks[0]
        self.k_t_r=ks[1]
        self.k_t_inf=ks[2]
        self.k_t_inc=ks[3]
        self.k_t_d=ks[4]
        
        self.recs, self.total, self.inf, self.dead, self.inc, self.dead_inc = data
        

    def loss(self, args):
        a_inf,b_inf,a_dead,b_dead,a_inc,b_inc= args
        
        k_inf=self.k_if
        teta_inf=self.teta_if
        k_dead=self.k_d
        teta_dead=self.teta_d
        k_inc=self.k_ic
        teta_inc=self.teta_ic

        inf_loss=0
        dead_loss=0
        inc_loss=0
        recs_loss=0
        total_loss=0
        if b_inf>=self.s*100 or b_dead>=self.s*100 or b_inc>=self.s*100 or b_inf<1 or b_dead<0 or b_inc<1:
            return max(abs(b_inf),abs(b_dead),abs(b_inc))*1e+290
        if (a_inf>len(self.inf) or a_dead>len(self.dead_inc) or a_inc>len(self.inc) 
              or a_inf<-(len(self.inf)/2) or a_dead<-(len(self.dead_inc)/2) or a_inc<-(len(self.inc)/2)):
            return max(abs(a_inf),abs(a_dead),abs(a_inc))*1e+290
        else:
            for i in range(len(self.recs)):
                inf_pred = max(0,(b_inf*((i-a_inf)**(k_inf-1)*
                                  ((math.e**(-(i-a_inf)/teta_inf))/(teta_inf**k_inf*gamma(k_inf))))).real)
                dead_pred = max(0,(b_dead*((i-a_dead)**(k_dead-1)*
                                  ((math.e**(-(i-a_dead)/teta_dead))/(teta_dead**k_dead*gamma(k_dead))))).real)
                inc_pred = max(0,(b_inc*((i-a_inc)**(k_inc-1)*
                                  ((math.e**(-(i-a_inc)/teta_inc))/(teta_inc**k_inc*gamma(k_inc))))).real)

                recs_pred = calculate_recs_gamma(i,a_inf,b_inf,k_inf,teta_inf,a_dead,b_dead,
                                           k_dead,teta_dead,a_inc,b_inc,k_inc,teta_inc)

                total_pred =0
                for j in range(i+1):
                    deads=0
                    for k in range(j+1):
                        deads+=max(0,(b_dead*((k-a_dead)**(k_dead-1)*
                                  ((math.e**(-(k-a_dead)/teta_dead))/(teta_dead**k_dead*gamma(k_dead))))).real)
                total_pred += (max(0,(b_inf*((i-a_inf)**(k_inf-1)*
                                  ((math.e**(-(i-a_inf)/teta_inf))/(teta_inf**k_inf*gamma(k_inf))))).real) + 
                               calculate_recs_gamma(i,a_inf,b_inf,k_inf,teta_inf,a_dead,b_dead,
                                           k_dead,teta_dead,a_inc,b_inc,k_inc,teta_inc)+
                              deads)


                inf_loss+=abs((self.inf[i] - inf_pred)*self.k_t_inf)
                dead_loss+=abs((self.dead_inc[i]-dead_pred)*self.k_t_d)
                inc_loss+=abs((self.inc[i]-inc_pred)*self.k_t_inc)
                recs_loss+=abs((self.recs[i] - recs_pred)*self.k_t_r)
                total_loss+=abs((self.total[i] - total_pred))
            return (inf_loss+dead_loss+inc_loss+recs_loss+total_loss)/5/self.s 
  
    
    def loss_gamma(self,args):
        a_inf,b_inf,k_inf,teta_inf,a_dead,b_dead,k_dead,teta_dead,a_inc,b_inc,k_inc,teta_inc= args
        inf_loss=0
        dead_loss=0
        inc_loss=0
        recs_loss=0
        total_loss=0
        if b_inf>=self.s*100 or b_dead>=self.s*100 or b_inc>=self.s*100 or b_inf<1 or b_dead<0 or b_inc<1:
            return max(abs(b_inf),abs(b_dead),abs(b_inc))*1e+290
        elif (a_inf>len(self.inf) or a_dead>len(self.dead_inc) or a_inc>len(self.inc) 
              or a_inf<-(len(self.inf)/2) or a_dead<-(len(self.dead_inc)/2) or a_inc<-(len(self.inc)/2)):
            return max(abs(a_inf),abs(a_dead),abs(a_inc))*1e+290
        else:
            for i in range(len(self.recs)):
                inf_pred = max(0,(b_inf*((i-a_inf)**(k_inf-1)*
                                  ((math.e**(-(i-a_inf)/teta_inf))/(teta_inf**k_inf*gamma(k_inf))))).real)
                dead_pred = max(0,(b_dead*((i-a_dead)**(k_dead-1)*
                                  ((math.e**(-(i-a_dead)/teta_dead))/(teta_dead**k_dead*gamma(k_dead))))).real)
                inc_pred = max(0,(b_inc*((i-a_inc)**(k_inc-1)*
                                  ((math.e**(-(i-a_inc)/teta_inc))/(teta_inc**k_inc*gamma(k_inc))))).real)
    
                recs_pred = calculate_recs_gamma(i,a_inf,b_inf,k_inf,teta_inf,a_dead,b_dead,
                                           k_dead,teta_dead,a_inc,b_inc,k_inc,teta_inc)
    
                total_pred =0
                for j in range(i+1):
                    deads=0
                    for k in range(j+1):
                        deads+=max(0,(b_dead*((k-a_dead)**(k_dead-1)*
                                  ((math.e**(-(k-a_dead)/teta_dead))/(teta_dead**k_dead*gamma(k_dead))))).real)
                total_pred += (max(0,(b_inf*((i-a_inf)**(k_inf-1)*
                                  ((math.e**(-(i-a_inf)/teta_inf))/(teta_inf**k_inf*gamma(k_inf))))).real) + 
                               calculate_recs_gamma(i,a_inf,b_inf,k_inf,teta_inf,a_dead,b_dead,
                                           k_dead,teta_dead,a_inc,b_inc,k_inc,teta_inc)+
                              deads)
    
    
                inf_loss+=abs((self.inf[i] - inf_pred)*self.k_t_inf)
                dead_loss+=abs((self.dead_inc[i]-dead_pred)*self.k_t_d)
                inc_loss+=abs((self.inc[i]-inc_pred)*self.k_t_inc)
                recs_loss+=abs((self.recs[i] - recs_pred)*self.k_t_r)
                total_loss+=abs((self.total[i] - total_pred))
            return (inf_loss+dead_loss+inc_loss+recs_loss+total_loss)/5/self.s





def load_data(name):
    if name=='Tatarstan':
        path = 'D:\\Covid-19\\data\\Crown_of_Tatarstan.xlsx'
        a = pd.read_excel(path)
        
        recs=[]
        total=[]
        inf=[]
        dead=[]
        inc=[]
        dead_inc=[]
        for i in range(55):
            recs.append(0)
            total.append(0)
            inf.append(0)
            dead.append(0)
        for i in range(len(a)):
            recs.append(int(a['Recovered'][i]))
            total.append(int(a['Total'][i]))
            dead.append(int(a['Deaths'][i]))
            inf.append(int(a['Total'][i]-a['Deaths'][i]-a['Recovered'][i]))
        
        dead_inc.append(dead[0])
        inc.append(total[0])
        for i in range(len(dead)-1):
            dead_inc.append(dead[i+1]-dead[i])
            inc.append(total[i+1]-total[i])
            
        return recs, total, inf, dead, inc, dead_inc
    else:
        response = requests.get('https://api.covid19api.com/country/'+name+'/status/confirmed/live')
        a=response.json()
        response = requests.get('https://api.covid19api.com/country/'+name+'/status/deaths/live')
        b=response.json()
        response = requests.get('https://api.covid19api.com/country/'+name+'/status/recovered/live')
        c=response.json()
        
        recs=[]
        total=[]
        inf=[]
        dead=[]
        inc=[]
        dead_inc=[]
        for i in range(len(a)):
            recs.append(c[i]['Cases'])
            total.append(a[i]['Cases'])
            dead.append(b[i]['Cases'])
            inf.append(a[i]['Cases']-b[i]['Cases']-c[i]['Cases'])
        recs=recs[:len(recs)-1]
        total=total[:len(total)-1]
        inf=inf[:len(inf)-1]
        dead=dead[:len(dead)-1]
        inc=inc[:len(inc)-1]
        dead_inc=dead_inc[:len(dead_inc)-1]
        
        dead_inc.append(dead[0])
        inc.append(total[0])
        for i in range(len(dead)-1):
            dead_inc.append(dead[i+1]-dead[i])
            inc.append(total[i+1]-total[i])
            
        return recs, total, inf, dead, inc, dead_inc
 
    
 
    
def optimize_country(name,t):
    global param
    
    data = load_data(name)
    s = sum(data[1])
    k_t_r=s/sum(data[0])
    k_t_inf=s/sum(data[2])
    k_t_inc=s/sum(data[4])
    k_t_d=s/sum(data[5])
    
    x0_gamma=param[name][t]
    
    if t=='Original':
        opt = Optimizer('Russia',[s,k_t_r,k_t_inf,k_t_inc,k_t_d],data)
        x_pred_gamma=optimize.minimize(opt.loss_gamma,x0_gamma, method='Nelder-Mead', tol=0.01)
        param[name][t]=x_pred_gamma.x.tolist()
    else:
        opt = Optimizer(t,[s,k_t_r,k_t_inf,k_t_inc,k_t_d],data)
        x_pred_gamma=optimize.minimize(opt.loss,x0_gamma, method='Nelder-Mead', tol=0.01)
        param[name][t]=x_pred_gamma.x.tolist()




        
def predict(n,results):
    n_f = np.zeros(shape=(n,))
    for i in range(len(n_f)):
        n_f[i] = max(0,(results[1]*((i-results[0])**(results[2]-1)*
                      ((math.e**(-(i-results[0])/results[3]))/(results[3]**
                                                                             results[2]*
                                                                 gamma(results[2]))))).real)

    n_d = np.zeros(shape=(n,))
    for i in range(n):
        n_d[i] = max(0,(results[5]*((i-results[4])**(results[6]-1)*
                      ((math.e**(-(i-results[4])/results[7]))/(results[7]**
                                                                             results[6]*
                                                                 gamma(results[6]))))).real)

    n_c = np.zeros(shape=(n,))
    for i in range(len(n_c)):
        n_c[i] = max(0,(results[9]*((i-results[8])**(results[10]-1)*
                      ((math.e**(-(i-results[8])/results[11]))/(results[11]**
                                                                              results[10]*
                                                                  gamma(results[10]))))).real)

    n_r = np.zeros(shape=(n,))
    for i in range(len(n_r)):
        n_r[i] = calculate_recs_gamma(i,results[0],results[1],results[2],results[3],
                                results[4],results[5],results[6],results[7],
                                results[8],results[9],results[10],results[11])

    n_t = np.zeros(shape=(n,))
    for i in range(len(n_t)):
        n_t[i]=0
        for j in range(i+1):
            deads=0
            for k in range(j+1):
                deads+=max(0,(results[5]*((i-results[4])**(results[6]-1)*
                      ((math.e**(-(i-results[4])/results[7]))/(results[7]**
                                                                             results[6]*
                                                                 gamma(results[6]))))).real)
        n_t[i] += (max(0,(results[1]*((i-results[0])**(results[2]-1)*
                      ((math.e**(-(i-results[0])/results[3]))/(results[3]**
                                                                             results[2]*
                                                                 gamma(results[2]))))).real) + 
                    calculate_recs_gamma(i,results[0],results[1],results[2],results[3],
                                results[4],results[5],results[6],results[7],
                                results[8],results[9],results[10],results[11])+
                    deads)
    return {'n_t':n_t.tolist(),'n_f':n_f.tolist(),'n_d':n_d.tolist(),
            'n_c':n_c.tolist(),'n_r':n_r.tolist()}




def main():
    global param
    
    j = {'Dates':[],
       'Countries':{
           'Russia':{
               'Predictions':{}
           },
           'Italy':{
               'Predictions':{}
           },
           'Tatarstan':{
               'Predictions':{}
           },
           'Iran':{
               'Predictions':{}
           },
           'Turkey':{
               'Predictions':{}
           },
           'Germany':{
               'Predictions':{}
           },
       }
    }
    
    ts1=[]
    for key in param.keys():
        t = threading.Thread(target=optimize_country, args=(key,'Original',))
        t.start()
        ts1.append(t)
        
    for t in ts1:
        t.join()
    
    ts2=[]
    for key in ['Russia','Tatarstan']:
        for t_key in param[key].keys():
            t = threading.Thread(target=optimize_country, args=(key,t_key,))
            t.start()
            ts2.append(t)
            
    for t in ts2:
        t.join()
        
    with open("D:\\Covid-19\\data\\param.json", "w") as write_file:
        json.dump(param, write_file)
    
    data=load_data('Russia')
    dates=[]
    x = [datetime.datetime(year=2020, month=1, day=22)]
    for i in range(len(data[0])+100):
        x.append(x[-1]+datetime.timedelta(days=1))
    for d in x:
        dates.append(d.strftime('%d-%b-%Y'))
    j['Dates']=dates
    
    for key in param.keys():
        data=load_data(key)
        j['Countries'][key]['Real_data']={'total':data[1],'dead_inc':data[5],'recs':data[0],
                                                  'dead':data[3],'inc':data[4],'inf':data[2]}
        for t_key in param[key].keys():
            p=[]
            if t_key=='Original':
                p=param[key][t_key]
            else:
                p=[param[key][t_key][0],param[key][t_key][1],param[t_key]['Original'][2],param[t_key]['Original'][3],
                   param[key][t_key][2],param[key][t_key][3],param[t_key]['Original'][6],param[t_key]['Original'][7],
                   param[key][t_key][4],param[key][t_key][5],param[t_key]['Original'][10],param[t_key]['Original'][11],
                  ]
            j['Countries'][key]['Predictions'][t_key]=predict(len(data[0])+100,np.array(p))
            
    with open("D:\\Covid-19\\data\\covid_file.json", "w") as write_file:
        json.dump(j, write_file)
        
main()