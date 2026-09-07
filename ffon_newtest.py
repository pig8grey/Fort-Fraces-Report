#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 08:30:00 2024

@author: jeff
"""


import os
import sys
import glob
import obspy
import obspy.signal.filter
import scipy
import gc
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import csv
from obspy import UTCDateTime, read_inventory,Stream
from datetime import datetime, timedelta, timezone
import matplotlib.dates as mdates
from multiprocessing import Pool
import pandas as pd
from meteostat import Hourly
from zoneinfo import ZoneInfo
from matplotlib.patches import Rectangle
from scipy.signal import lfilter,detrend
from astral.sun import sun
from astral import LocationInfo
from collections import deque
from obspy.signal.filter import envelope
from cutoffs import generalCutout,specialCutout
import heapq
from scipy.signal import hilbert
global peak_freqs

peak_freqs=[]
np.set_printoptions(legacy='1.25')
latitude = 48.61527780
longitude = -93.40166670

matplotlib.use('agg')
res=(1100,1250)
global currentStress
currentStress=0
# strainInfo={
# '082BH1':'South-North-Bottom',
# '082BH2':'North-Norht-Bottom',
# '082BH3':'North-top',
# '080BH3':'Southtop',
# '080BH2':'North-Southbottom',
# '080BH1':'South-Southbottom',
# }
stID={'080':'s','082':'n'}
strainInfo={
'082BH1':'N_GDR_BF_INS',
'082BH2':'N_GDR_BF_OUT',
'082BH3':'N_GDR_TF',


'080BH3':'S_GDR_TF',
'080BH2':'S_GDR_BF_INS',
'080BH1':'S_GDR_BF_OUT',
}


tz=7

mytime=UTCDateTime(2025,8,12)
l = LocationInfo('Fort Frances', 'Ontario', 'America/Winnipeg', latitude, longitude)
sunt = sun(l.observer, date=mytime, tzinfo=l.timezone)
# mytime=datetime.now()

myZone=ZoneInfo("America/Winnipeg")


# mytime=datetime(2022,10,6,0,0,0)
# mytime=datetime(2024,2,27,0,0,0)
myyear='{}'.format(mytime.year)
mymonth=f'{mytime.month:02}'
myday=f'{mytime.day:02}'


nexttime=mytime+timedelta(days=1)
myyear2='{}'.format(nexttime.year)
mymonth2=f'{nexttime.month:02}'
myday2=f'{nexttime.day:02}'

ws=datetime(mytime.year,mytime.month,mytime.day)


weatherstart=datetime(ws.year,ws.month,ws.day,7,0)
weather = Hourly('71962', weatherstart,
                 datetime(nexttime.year,nexttime.month,nexttime.day,6,59))
weather = weather.fetch()
weather = weather.temp
weather.index=weather.index-timedelta(hours=5)

print(myyear,mymonth,myday)


df=pd.read_csv(f'/home/jeff/ffon/S5_{myyear}-{mymonth}-{myday}_logs.csv',sep=',',index_col=False)

df=df[df.notna().all(1)]


pt=df.pivot_table(index=['Start Date (CST)','Start Time (CST)'],              
columns=['Box ID',], values=['Z(+ve)', 'Z(-ve)','Longitudinal(+ve)','Longitudinal(-ve)',
        'Lateral(+ve)','Lateral(-ve)'])


        
pt=pt[pt.notna().all(1)]
dates=pt.index.values
evtend=[]
anend=[]


for i,v in enumerate(pt.index.values):
    asd=df[df['Start Time (CST)']==pt.index.values[i][-1]]
    mt=max(asd['End Time (CST)'])
    evtend.append(UTCDateTime(mt))




    
cond = df['Start Time (CST)'].isin(pt.index.get_level_values(1))
df=df.drop(df[cond].index, inplace = False)
    

temp_df=df.drop(df[cond].index, inplace = False)

mydate=[UTCDateTime('{}T{}'.format(d[0],d[1])) for d in dates]
antime=[UTCDateTime(f'{myyear}/{mymonth}/{myday}T{currenttime}') for currenttime in temp_df['Start Time (CST)'].to_list()]
anend=[UTCDateTime(currenttime) for currenttime in temp_df['End Time (CST)'].to_list()]
# antime=antime[0:4]
# anend=anend[0:4]

sunFactor={'night':[0.3,0.3,0.3],
           'dawn':[0.1,0.1,0.1],
          'sunrise':[1.1,1.1,1.1],
          'noon':[0.5,0.5,0.5],
          'sunset':[0.3,0.3,0.3],
          'dusk':[0.1,0.1,1.1]}
print(mydate,antime)

z_values = pt[['Z(+ve)', 'Z(-ve)']]
z_diff = pt['Z(+ve)'] - pt['Z(-ve)']
z_diff = z_diff.values

class CNPlotter():
    
    __slots__='files2','files','stfiles','stfiles2','disst','stst','vst','vfiles','vfiles2'
    def __init__(self,mytime):
        self.files2=['/home/jeff/displacement/NBTM/DIS_NBTM_S5_{}_{}_{}.mseed'.format(
                    myyear2,
                    mymonth2,
                    myday2,),
                  '/home/jeff/displacement/SBTM/DIS_SBTM_S5_{}_{}_{}.mseed'.format(
                            myyear2,
                            mymonth2,
                            myday2,),   
            ] 
        self.files=['/home/jeff/displacement/NBTM/DIS_NBTM_S5_{}_{}_{}.mseed'.format(
                    myyear,
                    mymonth,
                    myday,),
              '/home/jeff/displacement/SBTM/DIS_SBTM_S5_{}_{}_{}.mseed'.format(
                        myyear,
                        mymonth,
                        myday,),    
            ] 
        self.vfiles=['/home/jeff/sy2020/general/fortFrances/velocity/NBTM/NBTM_S5_{}_{}_{}.mseed'.format(
                    myyear,
                    mymonth,
                    myday,),
          '/home/jeff/sy2020/general/fortFrances/velocity/SBTM/SBTM_S5_{}_{}_{}.mseed'.format(
                    myyear,
                    mymonth,
                    myday,),   

            ] 


        self.vfiles2=['/home/jeff/sy2020/general/fortFrances/velocity/NBTM/NBTM_S5_{}_{}_{}.mseed'.format(
                    myyear2,
                    mymonth2,
                    myday2,),
          '/home/jeff/sy2020/general/fortFrances/velocity/SBTM/SBTM_S5_{}_{}_{}.mseed'.format(
                    myyear2,
                    mymonth2,
                    myday2,),   

            ] 

        self.stfiles=[
            
            '/home/jeff/sy2020/general/fortFrances/strain/080/080_S5_{}_{}_{}.mseed'.format(
                      myyear,
                      mymonth,
                      myday,),  
            '/home/jeff/sy2020/general/fortFrances/strain/082/082_S5_{}_{}_{}.mseed'.format(
                    myyear,
                    mymonth,
                    myday,),

          ]

        self.stfiles2=[
            

          '/home/jeff/sy2020/general/fortFrances/strain/080/080_S5_{}_{}_{}.mseed'.format(
                    myyear2,
                    mymonth2,
                    myday2,),
            '/home/jeff/sy2020/general/fortFrances/strain/082/082_S5_{}_{}_{}.mseed'.format(
                    myyear2,
                    mymonth2,
                    myday2,),
          ]
        
        self.disst=obspy.Stream()
        self.vst=obspy.Stream()
        self.stst=obspy.Stream()
        
            
            
    def plotSt2(self,st,debug=False):
        daily=False
        if st:
            if debug: 
                est=st.copy()
                est.filter('lowpass',freq=1.0)
            
            start_date = st[0].stats.starttime
            sampling_rate = st[0].stats.sampling_rate
            print()
            if st[0].stats.npts>3*3600*1000:daily=True
            # Calculate time values
            sensor=st[0].stats.station
            plt.rcParams['figure.dpi'] = 80
            plt.rcParams['figure.figsize'] = [8.0, 9.0]
            # Create subplots
            
            fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.0, 9.0))
            
            # Plot the first waveform
            axes[0].plot(st.select(channel='B*Z')[0].times(type="matplotlib"), 
                         st.select(channel='B*Z')[0].data, label=f'{sensor}\nVertical')
            if debug:
                axes[0].plot(est.select(channel='B*Z')[0].times(type="matplotlib"), 
                             envelope(est.select(channel='B*Z')[0].data), label=f'{sensor}\nVertical')
            axes[0].set_ylabel('Displacement (mm)')
            axes[0].legend(loc="upper left")

            if len(st)>1:
                # Plot the second waveform
                if (len(st.select(channel='*N'))):
                    axes[1].plot(st.select(channel='B*N')[0].times(type="matplotlib"), 
                                 st.select(channel='B*N')[0].data, label=f'{sensor}\nLongitudonal')
                    axes[1].set_ylabel('Displacement (mm)')
                    axes[1].legend(loc="upper left")
                    
                    # Plot the third waveform
                if (len(st.select(channel='*E'))):
                    axes[2].plot(st.select(channel='B*E')[0].times(type="matplotlib"), 
                                 st.select(channel='B*E')[0].data, label=f'{sensor}\nLateral')
                    axes[2].set_xlabel('Time (CST)')
                    axes[2].set_ylabel('Displacement (mm)')
            
            if st.select(channel='B*E')[0].stats.npts>3600*1000:
                axes[2].xaxis.set_major_locator(mdates.HourLocator(interval=3)) 
            elif st.select(channel='B*E')[0].stats.npts>1200*1000:
                    axes[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=5)) # Set ticks every 6 hours
            else:
                axes[2].xaxis.set_major_locator(mdates.MinuteLocator(interval=2)) 
                
            axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[2].legend(loc="upper left")
            
            if debug:
                axes[1].plot(est.select(channel='B*N')[0].times(type="matplotlib"), 
                             envelope(est.select(channel='B*N')[0].data), label=f'{sensor}\nVertical')
                axes[2].plot(st.select(channel='B*E')[0].times(type="matplotlib"), 
                             envelope(est.select(channel='B*E')[0].data), label=f'{sensor}\nVertical')    
                
            if daily:
                for begin_time,end_time in zip(mydate,evtend):
                    
                    begin_time=begin_time-timedelta(minutes=5)
                    end_time=end_time+timedelta(minutes=5)
                    for i,ax in enumerate(axes):
                        axes[i].grid('on', linestyle='--')
                        axes[i].add_patch(Rectangle((mdates.date2num(begin_time), axes[i].get_ylim()[0]),
                                       mdates.date2num(end_time) - mdates.date2num(begin_time),
                                       axes[i].get_ylim()[1] - axes[i].get_ylim()[0],
                                       alpha=0.4, color='green'))
                        
                    
            # for begin_time,end_time in zip(antime,anend):
            #     begin_time=begin_time-timedelta(minutes=5)
            #     end_time=end_time+timedelta(minutes=5)
            #     for i,ax in enumerate(axes):
            #         axes[i].add_patch(Rectangle((mdates.date2num(begin_time), axes[i].get_ylim()[0]),
            #                        mdates.date2num(end_time) - mdates.date2num(begin_time),
            #                        axes[i].get_ylim()[1] - axes[i].get_ylim()[0],
            #                        alpha=0.4, color='red'))
                    

            # Customize the overall plot
            plotname=st[0].stats.starttime.strftime('%Y%m%d%H%M')
            plt.subplots_adjust(hspace=0, wspace=0)  # Adjust the right margin as needed
            plt.tight_layout()  # Increase padding between subplots
            fig.savefig(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{sensor}-{plotname}_test.png')
            plt.clf()
            plt.close(fig)
            gc.collect()
            return
        
    def plotSt3(self,st,debug=False):
            daily=False
            if st:
                if debug: 
                    est=st.copy()
                    est.filter('lowpass',freq=1.0)
                
                start_date = st[0].stats.starttime
                sampling_rate = st[0].stats.sampling_rate
                print()
                if st[0].stats.npts>3*3600*1000:daily=True
                # Calculate time values
                sensor=st[0].stats.station
                plt.rcParams['figure.dpi'] = 80
                plt.rcParams['figure.figsize'] = [8.0, 3.0]
                # Create subplots
                
                fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(8.0, 3.0))
                
                # Plot the first waveform
                axes.plot(st.select(channel='B*3')[0].times(type="matplotlib"), 
                             st.select(channel='B*3')[0].data, label=f'{sensor}\nVertical')
                if debug:
                    axes.plot(est.select(channel='B*Z')[0].times(type="matplotlib"), 
                                 envelope(est.select(channel='B*Z')[0].data), label=f'{sensor}\nVertical')
                axes.set_ylabel('Displacement (mm)')
                axes.legend(loc="upper left")

             
                
                if st[0].stats.npts>3600*1000:
                    axes.xaxis.set_major_locator(mdates.HourLocator(interval=3)) 
                elif st[0].stats.npts>1200*1000:
                        axes.xaxis.set_major_locator(mdates.MinuteLocator(interval=5)) # Set ticks every 6 hours
                else:
                    axes.xaxis.set_major_locator(mdates.MinuteLocator(interval=2)) 
                    
                axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                axes.legend(loc="upper left")
                
                    
                if daily:
                    for begin_time,end_time in zip(mydate,evtend):
                        
                        begin_time=begin_time-timedelta(minutes=5)
                        end_time=end_time+timedelta(minutes=5)
                        for i,ax in enumerate(axes):
                            axes[i].grid('on', linestyle='--')
                            axes[i].add_patch(Rectangle((mdates.date2num(begin_time), axes[i].get_ylim()[0]),
                                           mdates.date2num(end_time) - mdates.date2num(begin_time),
                                           axes[i].get_ylim()[1] - axes[i].get_ylim()[0],
                                           alpha=0.4, color='green'))
                            
                        
                # for begin_time,end_time in zip(antime,anend):
                #     begin_time=begin_time-timedelta(minutes=5)
                #     end_time=end_time+timedelta(minutes=5)
                #     for i,ax in enumerate(axes):
                #         axes[i].add_patch(Rectangle((mdates.date2num(begin_time), axes[i].get_ylim()[0]),
                #                        mdates.date2num(end_time) - mdates.date2num(begin_time),
                #                        axes[i].get_ylim()[1] - axes[i].get_ylim()[0],
                #                        alpha=0.4, color='red'))
                        

                # Customize the overall plot
                plotname=st[0].stats.starttime.strftime('%Y%m%d%H%M')
                plt.subplots_adjust(hspace=0, wspace=0)  # Adjust the right margin as needed
                plt.tight_layout()  # Increase padding between subplots
                fig.savefig(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{sensor}-{plotname}_test.png')
                plt.clf()
                plt.close(fig)
                gc.collect()
                return
        
        
    def filter_AC(self):
        for i,st in enumerate(self.stst):
            if st:
                self.stst[i]=st.filter('lowpass',freq=35.0)
                self.stst[i]=st.filter('lowpass',freq=35.0)
                if(st[i].stats.starttime+timedelta(minutes=3)<st[i].stats.endtime):
                    self.stst[i]=st.slice(st[i].stats.starttime+timedelta(minutes=3),st[i].stats.endtime)
            

            
    # def convE(self,st):
    #     Gf=2.08
    #     Rg=130.0
    #     Rl=0.5
    #     Vex=5.0
    #     gain=16.0
    #     result=st.copy()
    #     for i,tr in enumerate(st):
    #         volt=-1.0*tr.data*2.5/2**31/gain
    #         result[i].data=(-4.0*(volt/Vex)/(2.08*(1.0+2.0*(volt/Vex))))*(1.0+Rl/Rg)
    #     return result
    
    def convE(self,st):
        Gf=2.08
        Rg=120.0
        Rl=0.5
        Vex=5.0
        gain=16.0
        result=st.copy()
        for i,tr in enumerate(st):
            volt=-1.0*tr.data*2.5/2**31/gain
            # result[i].data=(-4.0*(volt/Vex)/(2.08*(1.0+2.0*(volt/Vex))))*(1.0+Rl/Rg)
            result[i].data=(-4.0/Gf)*(volt/Vex)
        return result    

    
    def plotDay(self):
        # for i,st in enumerate(self.disst):self.plotSt2(st) 
        with Pool(processes=4) as pool:
            pool.map(self.plotSt2,self.disst)
        return
            
    def getStress(self):
        self.filter_AC()
        with Pool(processes=4) as pool:
            self.stst=pool.map(self.convE,self.stst)
        # for i,st in enumerate(self.stst):self.stst[i]=self.convE(st)
        
        return self.stst
            

            
cn=CNPlotter(mytime)     

def find_time_of_day(target_datetime):
    """Find the time of day based on the provided datetime."""
    
    if sunt['dawn'] <= target_datetime < sunt['sunrise']:
        return "dawn"
    elif sunt['sunrise'] <= target_datetime < sunt['noon']:
        return "sunrise"
    elif sunt['noon'] <= target_datetime < sunt['sunset']:
        return "noon"
    elif sunt['sunset'] <= target_datetime < sunt['dusk']:
        return "sunset"
    elif sunt['dusk'] <= target_datetime:
        return "dusk"
    else:
        return "night"
    
    
def getEnvelope(info):
    
    st=info[0]
    if st:
        mytimes=info[1]
        start_time=st[0].stats.starttime
        end_time=st[0].stats.endtime
        moved_st=Stream()
        moved_st+=st
        j=0
        rate=2.0 
        # for i,tr in enumerate(st):
        #     if (tr.stats.channel=='BHZ' or tr.stats.channel=='BSZ'):
        #         data_envelope = envelope(tr.data) 
        #         moved_st[i].data= (-tr.data - data_envelope*rate)

        for i,evttimes in enumerate(mytimes):
            begin=evttimes[0]
            finish=evttimes[1]
            
            a=st.slice(begin-timedelta(minutes=1),
                       finish+timedelta(minutes=1))
            st=st.cutout(begin-timedelta(minutes=1),
                       finish+timedelta(minutes=1))            
            envst=a.copy()
            envst.filter('lowpass',freq=1.0)
            for j,tr in enumerate(a):
                if (tr.stats.channel=='BHZ' or tr.stats.channel=='BSZ'):
                    data_envelope = envelope(envst.select(channel='*Z')[0].data)  
                    print(rate)
                    # print(st,begin,finish)
                    a[j].data = (a[j].data - data_envelope*rate)
                    a[j].data = a[j].data.astype(np.float32)
                    st += a
                    st.merge(method=0,fill_value=0)
        return st

def getEnvelope2(info):
    
    st=info[0]
    if st:
        mytimes=info[1]
        myfreqs=info[2]
        start_time=st[0].stats.starttime
        end_time=st[0].stats.endtime
        moved_st=Stream()
        moved_st+=st
        j=0
        # for i,tr in enumerate(st):
        #     if (tr.stats.channel=='BHZ' or tr.stats.channel=='BSZ'):
        #         data_envelope = envelope(tr.data) 
        #         moved_st[i].data= (-tr.data - data_envelope*rate)

        for i,evttimes in enumerate(mytimes):
            begin=evttimes[0]
            finish=evttimes[1]
            
            a=st.slice(begin-timedelta(minutes=1),
                       finish+timedelta(minutes=1))
            st=st.cutout(begin-timedelta(minutes=1),
                       finish+timedelta(minutes=1))            
            envst=a.copy()
            envst.filter('lowpass',freq=0.3)
            for j,tr in enumerate(a):
                if (tr.stats.channel=='BHZ' or tr.stats.channel=='BSZ'):
                    data_envelope = envelope(envst.select(channel='*Z')[0].data) 
                    start = a[j].stats.starttime
                    end = a[j].stats.endtime
                    window_length = 60.0  # seconds
                    current = start
                    k=0
                    myminutes=0
                    myind=0
                    while current + window_length <= end:
                        tr_slice = a[j].slice(current, current + window_length)
                        print(myind,myind+window_length*1000)
                        env_slice = data_envelope[int(myind):int(myind+window_length*1000)+1]
                        if 0.1<myfreqs[i][k]<0.2:
                                                        
                            if myminutes>3:
                                tr_slice.data = (tr_slice.data - env_slice*2.0)
                                
                            elif myminutes>1:
                                tr_slice.data = (tr_slice.data - env_slice*2.0)
                            else:
                                tr_slice.data = (tr_slice.data*1.2 - env_slice*3.2)
                                
                            myminutes+=1
                            
                        elif 0.2<myfreqs[i][k]<0.3:
                            tr_slice.data = (tr_slice.data*0.8 - env_slice*1.5)
                            myminutes=0
                        else:
                            tr_slice.data = (tr_slice.data*1.0 - env_slice*1.5)
                            myminutes=0
                    
                        current += window_length
                        k+=1
                        myind= myind+ 1000 * window_length
                        tr_slice.data=tr_slice.data.astype(np.float32)
                        st += tr_slice
                        continue
                    # print(st,begin,finish)
                    
                    st +=tr_slice
                    st.merge(method=0,fill_value=0)
        return st

    
    
def changeTime(st,myZone=ZoneInfo("America/Winnipeg")):
    for i,tr in enumerate(st):
        t=tr.stats.starttime.datetime.replace(tzinfo=timezone.utc)
        st[i].stats.starttime=UTCDateTime(t.astimezone(myZone).strftime("%Y%m%d%H%M%S"))
    return st



        
def corrSB(data,prev):
    
    x1,x2,xx1,xx2,xxx1,xxx2,xxxx1,xxxx2=prev

    nump=[1,-2,1]
    denp=[1,-1.999912035,0.99991204]
        
    num1=[1.0,-1.914500,0.918283]
    den1=[1.0,-1.99808,0.998084]  
    

    
    num2=[1,-1.999996845429212, 0.999996845429212]
    den2=[1,-1.996481293746107,0.996487599189977]        
    
    [data,x12]=lfilter(num1,den1,data,zi=[x1,x2])
    x1,x2=x12
    [data,xx12]=lfilter(nump,denp,data,zi=[xx1,xx2])
    xx1,xx2=xx12
    data=np.cumsum(data)
    [data,xxx12]=lfilter(num2,den2,data,zi=[xxx1,xxx2])
    xxx1,xxx2=xxx12
    # [data,xxxx12]=lfilter(num2,den2,data,zi=[xxxx1,xxxx2])
    # xxxx1,xxxx2=xxxx12
    data=data/2**31*2.5/20.5
    

    return [x1,x2,xx1,xx2,xxx1,xxx2,data]



    
def gettime(path):

    combtime=path.split('.')[0]


    if sys.platform.startswith('win'):
        checklen=combtime.split('\\')
    else:
        checklen=combtime.split('/')
    

    if len(checklen)<2:
        checklen = combtime.split('/')


    try:    
        if len(checklen[-1])>5:
            a=[s for s in checklen[-1].split('_') if s.isdigit()]
            starttime=datetime.strptime(a[0], '%Y%m%d%H%M%S')

            # ID=checklen[-2]
        else:
            dirs=path.split('/')
            
            if len(dirs)<2:
                dirs=path.split('\\')
            hms=dirs[-2]
            ymd=dirs[-3]
            hd=dirs[-1].split('.')[0]
            year=2000+int(ymd[0:2])
            month=int(ymd[2:4])
            day=int(ymd[4:])
            hour=int(hms[0:2])
            minute=int(hms[2:4])
            second=int(hms[4:])
            
            starttime=datetime(year,month,day,hour,minute,second)
            
            starttime=starttime+timedelta(hours=int(hd))


            # ID=checklen[-4]

       
        return starttime
    except:
        return 0
    

def OriginalCN(data,prev):
    
    x1,x2,xx1,xx2,xxx1,xxx2,xxxx1,xxxx2=prev

    nump=[1,-2,1]
    denp=[1,-1.999912035,0.99991204]
        
    num1=[1.0,-1.914500,0.918283]
    den1=[1.0,-1.99808,0.998084]  
    

    
    num2=[1,-2,1]
    den2=[1,-1.9991203,0.9991207]          
    
    [data,x12]=lfilter(num1,den1,data,zi=[x1,x2])
    x1,x2=x12
    [data,xx12]=lfilter(nump,denp,data,zi=[xx1,xx2])
    xx1,xx2=xx12
    data=np.cumsum(data)
    [data,xxx12]=lfilter(num2,den2,data,zi=[xxx1,xxx2])
    xxx1,xxx2=xxx12
    [data,xxxx12]=lfilter(num2,den2,data,zi=[xxxx1,xxxx2])
    xxxx1,xxxx2=xxxx12
    data=data/2**31*2.5/20.5
    

    return [x1,x2,xx1,xx2,xxx1,xxx2,data]



def downsample(factor,values):
    buffer_ = deque([],maxlen=factor)
    downsampled_values = []
    for i,value in enumerate(values):
        buffer_.appendleft(value)
        if (i-1)%factor==0:
            #Take max value out of buffer
            # or you can take higher value if their difference is too big, otherwise just average
            downsampled_values.append(max(max(buffer_), min(buffer_), key=abs))
    return np.array(downsampled_values)


def readStream(mypath):
    a=Stream()
    if os.path.isfile(mypath[0]):a=obspy.read(mypath[0])    
    if os.path.isfile(mypath[1]):a.extend(obspy.read(mypath[1]))
    a=a.trim(UTCDateTime(f'{myyear}-{mymonth}-{myday}T05:00'),
             UTCDateTime(f'{myyear2}-{mymonth2}-{myday2}T06:29:59.999'))
    return a

def format_number(value):
    # Format the number to have at most 5 decimal places
    return "{:.7f}".format(value)




def plotWeather(df):
    plt.clf()
    plt.rcParams['figure.figsize'] = [8.0, 9.0]
    plt.figure().set_figheight(9)
    plt.rcParams['figure.dpi'] = 80
    myFmt = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(myFmt)

    # plt.title(f'{myyear}-{mymonth}-{myday}, Hourly Temperature °C')
    plt.plot_date(df.index.to_numpy(),df.values,'-')
    plt.ylabel('Temperature °C')
    plt.xlabel('Time')
    plt.savefig(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{myyear}-{mymonth}-{myday}_temperature.png')
    return
    

def writeStraincsv(data,filename,mode='a'):
    print(data)
    file_exists = os.path.exists(filename)
    if data[2]=='080':
        header=['Start Date (CST)','Start Time (CST)',
                   'Box ID','S_GDR_BF_OUT(Max)','S_GDR_BF_INS(Max)','S_GDR_TF(Max)','S_GDR_BF_OUT(Min)',
                   'S_GDR_BF_INS(Min)','S_GDR_TF(Min)','End Time (CST)']
    if data[2]=='082':
        header=['Start Date (CST)','Start Time (CST)',
                   'Box ID','N_GDR_BF_INS(Max)','N_GDR_BF_OUT(Max)','N_GDR_TF(Max)',
                   'N_GDR_BF_INS(Min)','N_GDR_BF_OUT(Min)','N_GDR_TF(Min)','End Time (CST)']
    with open(filename, mode, newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        if header and mode=='w':
            csv_writer.writerow(header)

        if isinstance(data, dict):
            formatted_data = [format_number(value) if isinstance(value, float) else value for value in data.values()]
            csv_writer.writerow(formatted_data)
        elif isinstance(data, list):
            formatted_data = [format_number(value) if isinstance(value, float) else value for value in data]
            csv_writer.writerow(formatted_data)

    return

def writeLogcsv(data,filename,mode='a'):
    print(data)
    file_exists = os.path.exists(filename)

    with open(filename, mode, newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        if isinstance(data, dict):
            formatted_data = [format_number(value) if isinstance(value, float) else value for value in data.values()]
            csv_writer.writerow(formatted_data)
        elif isinstance(data, list):
            formatted_data = [format_number(value) if isinstance(value, float) else value for value in data]
            csv_writer.writerow(formatted_data)

    return


def plotEvt(st, begin=mydate,finish=evtend,isShow=False,res=res):

    for i,funct in enumerate(begin):
        print(funct)
        a=st.slice(funct-timedelta(minutes=2),
                   finish[i]+timedelta(minutes=2))
        # print(a)
        if any(a):
            t=a[0].stats.starttime
            cn.plotSt3(a)
            # a.select(sampling_rate=1000).plot(outfile='/home/jeff/ffon/web/export/{}_{}_{}/{}-{}.png'.format(
            #     t.year,
            #     str(t.month).zfill(2),
            #     str(t.day).zfill(2),
            #     st[0].stats.station,
            #     (funct-timedelta(minutes=2)).strftime('%Y%m%d%H%M')),
            #     show=isShow,size=res,equal_scale=False)
        else: print('Empty Stream')
    return

def plotAno(st, begin=antime, finish=anend,isShow=False,res=res):
    for i,funct in enumerate(begin):

        a=st.slice(funct,
                   finish[i]+timedelta(minutes=2))
        # print(a)
        print(a)
        if any(a):
           cn.plotSt2(a)  
    

            
def plotEvtst(st,begin=mydate,finish=evtend,isShow=False,res=res,customFactor=[0,0,0]):
    strain=st.select(channel='BH*')
    # dis=st.select(channel='BS*')

    myMaxMin=[]
    
    for i,funct in enumerate(begin):
        
        a=strain.slice(funct-timedelta(minutes=2),
                   finish[i]+timedelta(minutes=2))
        sun_stat=find_time_of_day(funct)
        
        if any(a):
            a=changetoStress(a,customFactor)
            values=getMaxMin(a,'s')
            # print(values[0][3],(values[0][6]),np.abs(values[0][8])-values[0][5])
            # print(np.max(np.abs(a.max())),np.max(np.abs(a.max()))>2500)
            

            # if ((values[0][3]-np.abs(values[0][6]))>100
            # and (values[0][4]-np.abs(values[0][7]))>100): 
            # and (np.abs(values[0][8])-values[0][5])>100):
            
  
                
                # if a[0].stats.station=='082':
                #     a[2].data[a[2].data>200]-=1.5*a[2].data[a[2].data>200]

            plotStrain(a, 'train',1)
                    
                    

        
                
            
        else: print('Empty Stream')
        if any(a):
            t=funct-timedelta(minutes=2)
            print(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{t.strftime("%Y%m%d%H%M")}_{stID[a[2].stats.station]}.mseed')
            a[2]=a[2].write(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{t.strftime("%Y%m%d%H%M")}_{stID[a[2].stats.station]}.mseed')
           
    return myMaxMin


def manualEvtst(st,begin=mydate,finish=evtend,isShow=False,res=res,customFactor=[0,0,0],
                tslice=[],channel=None,offset=None):
    strain=st.select(channel='BH*')
    # dis=st.select(channel='BS*')
    
    myMaxMin=[]
    
    for i,funct in enumerate(begin):
        
        a=strain.slice(funct-timedelta(minutes=2),
                   finish[i]+timedelta(minutes=2))
        sun_stat=find_time_of_day(funct)
        
        if any(a):
            a=changetoStress(a,customFactor)
            

            if (any (tslice) or channel):
                if isinstance (channel,list):
                    for i,v in enumerate(a):
                        if (any (tslice)):
                            a[i]=a[i].slice(tslice[0],tslice[1])
                        a[i].data=a[i].data-offset[i]
                        
                else: 
                    if (any (tslice)):
                        a[channel-1]=a[channel-1].slice(tslice[0],tslice[1])
                    a[channel-1].data=a[channel-1].data-offset
 
                
            values=getMaxMin(a,'s')
            # print(values[0][3],(values[0][6]),np.abs(values[0][8])-values[0][5])
            # print(np.max(np.abs(a.max())),np.max(np.abs(a.max()))>2500)
            
            # for i,tr in enumerate(a):a[i].data=detrend(a[i].data,type='linear')
            # plotStrain(a,'train',1)
            # values=getMaxMin(a,'s')
            
            print(values)

            
            if(np.max(np.abs(a.max()))>2500.0):
                

                values=getMaxMin(a,'s')
                myMaxMin.append(values)
                plotStrain(a, 'train',1)
                
              
            else:plotStrain(a, 'other',1)
            

        
                
            
        else: print('Empty Stream')
        if any(a):
            t=funct-timedelta(minutes=2)
            a[2]=a[2].write(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{t.strftime("%Y%m%d%H%M")}_{stID[a[2].stats.station]}.mseed')
    return myMaxMin

def getValues(st,begin=mydate,finish=evtend,isShow=False,res=res):

    dis=st
    
    myMaxMin=[]

    for i,funct in enumerate(begin):
        print(funct)

        a=dis.slice(funct-timedelta(seconds=60),
                   finish[i]+timedelta(minutes=2))
        # b=getEnvelope(a)
        # print(a)
        if any(a):
            myMaxMin.append(getMaxMin(a,'d'))
                
    return myMaxMin

def removeLinear(tr):
    
    sun_factor=sunFactor[find_time_of_day(tr.stats.starttime)]
    x = np.arange(tr.stats.npts)
    try:
        y0, y1 = tr.data[0], tr.data[59999]
        slope = (y1 - y0) / (59999-1)
    except:
        y0, y1 = tr.data[0], tr.data[-1]
        slope = (y1 - y0) / (len(tr.data)-1)
    
    print('Slope=',slope)
    
    
    if np.abs(slope)>0.002:
        trend = 2.0*slope * x * sun_factor[0]
    elif np.abs(slope)>0.001:
        trend = 0.2*slope * x * sun_factor[1]
    else:
        trend = 1.2*slope * x * sun_factor[2]
    
    return trend

def customLinear(tr,factor=0.0):
    x = np.arange(tr.stats.npts)
    y0, y1 = tr.data[0], tr.data[59999]
    slope = (y1 - y0) / (59999-1)
    print('Slope=',slope)
    
    
    trend = factor*slope * x

    
    return trend


def apply_shape_transfer(target_stream, template_stream, target_chan='*Z', template_chan='*2', window_sec=0.5, force_p2p=None):
    """
    Modifies the target_stream in-place by applying the shape of template_stream
    onto the texture of target_stream.
    """
    # 1. Select the specific traces
    # copy() ensures we don't mess up the original raw data until we are ready
    tr_target = target_stream.select(channel=target_chan)[0].copy()
    tr_template=template_stream[0]
    # 2. CRITICAL: Resample Template to match Target
    # We interpolate the template to the exact sampling rate and start time of the target
    # so the arrays are the exact same length and time-aligned.
    tr_template.interpolate(sampling_rate=tr_target.stats.sampling_rate, 
                            starttime=tr_target.stats.starttime,
                            npts=tr_target.stats.npts)
    
    # 3. Extract Data as Pandas Series (for easy rolling mean)
    # We use the target's sampling rate to convert window_sec into samples
    window_samples = int(window_sec * tr_target.stats.sampling_rate)
    
    target_series = pd.Series(tr_target.data)
    template_series = pd.Series(tr_template.data)

    # 4. Decompose Signals (Trend vs. Texture)
    # Get the smooth trend (the "Shape")
    target_trend = target_series.rolling(window=window_samples, center=True, min_periods=1).mean()
    template_trend = template_series.rolling(window=window_samples, center=True, min_periods=1).mean()
    
    # Get the texture (the "Vibration") from the target
    target_texture = target_series - target_trend

    # 5. Scale Template Shape to Match Target Amplitude
    # Calculate Peak-to-Peak (P2P)
    if force_p2p is not None:
        target_p2p = force_p2p  # User overrides value (e.g., 1.5 or 1.9)
    else:
        # Automatic robust range (using quantiles is safer than max-min for noisy data)
        target_p2p = target_trend.max() - target_trend.min()
    template_p2p = template_trend.max() - template_trend.min()
    
    # Avoid division by zero
    scale_factor = (target_p2p / template_p2p) if template_p2p != 0 else 0
    
    # Scale the template
    scaled_template_shape = template_trend * scale_factor
    
    # Align Baseline: Shift the new shape so it starts roughly where the old one did
    # (Or align based on max/min depending on your preference)
    offset = target_trend.iloc[0] - scaled_template_shape.iloc[0]
    scaled_template_shape += offset

    # 6. Recombine: New Shape + Old Texture
    new_data_series = scaled_template_shape * 2.0 + target_texture
    
    # 7. Write back to a NEW Trace/Stream object
    tr_modified = tr_target.copy()
    tr_modified.data = np.array(new_data_series)  # Convert back to numpy
    
    # Update Stats to reflect modification (optional but good practice)
    tr_modified.stats.comment = "Modified with shape transfer from Strain Gauge"
    
    tr_modified.data=tr_modified.data.astype(np.float32)
    return tr_modified

def execute_auto_calibration(strain, displacement):

    strain_trace = strain[0].copy()
    
    # BUG 1 FIX: .select() returns a Stream. We need [0] to extract the actual Trace.
    displacement_trace = displacement.select(channel='*Z')[0].copy()
    # Step 1: Align arrays inside identical frequency window
    
    
    strain_trace.filter('highpass', freq=0.3)
    displacement_trace.filter('highpass', freq=0.3)

    print(strain_trace.max(), displacement_trace.max())
    st_data = strain_trace.data
    dis_data = displacement_trace.data
    
    st_data = np.abs(hilbert(st_data))
    dis_data = np.abs(hilbert(dis_data))
    # BUG 2 FIX: Use .data for BOTH arrays in the multiplication
    numerator = np.sum(st_data * dis_data)
    denominator = np.sum(st_data ** 2)
    if denominator == 0:
        raise ValueError("Mathematical failure: Division by zero in strain variance matrix.")
    calibration_coefficient = numerator / denominator
    return calibration_coefficient

def NewCN(data,prev):
    
    x1,x2,xx1,xx2,xxx1,xxx2,xxxx1,xxxx2=prev

    nump=[1.000000000000000,-1.919858511032698,0.923652405829430]
    denp=[1.0000000000000, -1.998077942846719, 0.998077942846720,]
        
    num1=[1.000000000000001,-1.999999950662830, 0.999999950662829]
    den1=[1.000000000000000,-1.999340384456179, 0.999340483119673]  
    

    
    num2=[1,-2,1]
    den2=[1,-1.9991203,0.9991207]          
    
    [data,x12]=lfilter(num1,den1,data,zi=[x1,x2])
    x1,x2=x12
    [data,xx12]=lfilter(nump,denp,data,zi=[xx1,xx2])
    xx1,xx2=xx12
    data=np.cumsum(data)
    [data,xxx12]=lfilter(num2,den2,data,zi=[xxx1,xxx2])
    xxx1,xxx2=xxx12
    [data,xxxx12]=lfilter(num2,den2,data,zi=[xxxx1,xxxx2])
    xxxx1,xxxx2=xxxx12
    data=data/2**31*2.5/20.5
    

    return [x1,x2,xx1,xx2,xxx1,xxx2,data]
def removeTrend(tr,CustomFactor=0.0):
    
    
    if CustomFactor:
        trend=customLinear(tr,CustomFactor)
    else:
        trend=removeLinear(tr)
    
    tr.data=tr.data-trend

    if tr.stats.channel=='BH3':
        tr.data=tr.data-np.max(tr.data)
    else:
        tr.data=tr.data-np.min(tr.data)
    return tr.data

def changetoStress(st,CustomFactor=[0,0,0]):
    print(st)
    for i,tr in enumerate(st):
        st[i].data=st[i].data*30000000.0
        st[i].data=detrend(st[i].data)
        if st[i].stats.station=='082' and i==2:
            st[i].data[st[i].data>50]-=1.5*st[i].data[st[i].data>50]
            st[i]=st[i].detrend('polynomial', order=2)
            st[i]=st[i].detrend('polynomial', order=2)
        # st[i].data=downsample(25, st[i].data)
        # st[i].stats.sampling_rate=40
        
        st[i]=st[i].slice(st[i].stats.starttime+timedelta(minutes=1),
                          st[i].stats.endtime)

            
        st[i].data=removeTrend(st[i],CustomFactor[i])
        try:
            st[i].data=st[i].data-st[i].data[1000]
        except:
            st[i].data=st[i].data-st[i].data[0]
    return st




def normalized_stress_0(x):
    return ((x - c0)*30000000.0)


def normalized_strain_0(x):
    return ((x + c0)/30000000.0 )

def normalized_stress_1(x):
    return ((x - c1)*30000000.0)


def normalized_strain_1(x):
    return ((x + c1)/30000000.0 )

def normalized_stress_2(x):
    return ((x - c2)*30000000.0)


def normalized_strain_2(x):
    return ((x + c2)/30000000.0 )


def plotStrain(st,plottype='',env=None):
    if st:
                
        global c0,c1,c2
        st=st.merge(method=0,fill_value='latest')

        start_date = st[0].stats.starttime
        sampling_rate = st[0].stats.sampling_rate

        # mytimes=
        
        sensor=st[0].stats.station
        plt.rcParams['figure.dpi'] = 80
        plt.rcParams['figure.figsize'] = [8.0, 9.0]
        # Create subplots

        
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.0, 9.0))
        # Plot the first waveform
        if st[0].stats.station=='080':
            axes[0].plot(st.select(channel='*1')[0].times(type="matplotlib"), 
                         st.select(channel='*1')[0].data, label=strainInfo[f'{sensor}{st[0].stats.channel}'])
        else:
            axes[0].plot(st.select(channel='*2')[0].times(type="matplotlib")
                         ,st.select(channel='*2')[0].data, label=strainInfo[f'{sensor}{st[1].stats.channel}'])
            
        if env:axes[0].set_ylabel('Stress(psi)')
        else:
            axes[0].set_ylabel('Strain')
            c0= np.min(st.select(channel='*1')[0].data)
            secax_y = axes[0].secondary_yaxis('right', 
                    functions=(normalized_stress_0, normalized_strain_0))
            
            secax_y.set_ylabel('Stress(psi)')
        axes[0].legend(loc="upper right")


        if len(st)>1:
                # Plot the second waveform
            if st[0].stats.station=='080':
                    axes[1].plot(st.select(channel='*2')[0].times(type="matplotlib")
                                 , st.select(channel='*2')[0].data, label=strainInfo[f'{sensor}{st[1].stats.channel}'])
            else:
                    axes[1].plot(st.select(channel='*1')[0].times(type="matplotlib")
                                 , st.select(channel='*1')[0].data, label=strainInfo[f'{sensor}{st[0].stats.channel}'])
            if env:
                axes[1].set_ylabel('Stress(psi)')
            else:
                axes[1].set_ylabel('Strain')
                
                if st[0].stats.station=='080':
                    c1= np.min(st.select(channel='*2')[0].data)
                    secax_y = axes[1].secondary_yaxis('right', 
                            functions=(normalized_stress_1, normalized_strain_1))
                    secax_y.set_ylabel('Stress(psi)')
                else:
                    c1= np.min(st.select(channel='*1')[0].data)
                    secax_y = axes[1].secondary_yaxis('right', 
                            functions=(normalized_stress_1, normalized_strain_1))
                    secax_y.set_ylabel('Stress(psi)')
            axes[1].legend(loc="upper right")
            
            # Plot the third waveform
            
            axes[2].plot(st.select(channel='*3')[0].times(type="matplotlib")
                         , st.select(channel='*3')[0].data, label=strainInfo[f'{sensor}{st[2].stats.channel}'])
            
            if env:axes[2].set_ylabel('Stress(psi)')
            else:
                axes[2].set_ylabel('Strain')
                c2= np.max(st.select(channel='*3')[0].data)
                secax_y = axes[2].secondary_yaxis('right', 
                        functions=(normalized_stress_2, normalized_strain_2))
                secax_y.set_ylabel('Stress(psi)')
            axes[2].legend(loc="upper right")
            axes[2].set_xlabel('Time (CST)')

            
        if not plottype:
            for begin_time,end_time in zip(mydate,evtend):
                
                begin_time=begin_time-timedelta(minutes=5)
                end_time=end_time+timedelta(minutes=5)
                for i,ax in enumerate(axes):
                    axes[i].grid('on', linestyle='--')
                    axes[i].add_patch(Rectangle((mdates.date2num(begin_time), axes[i].get_ylim()[0]),
                                   mdates.date2num(end_time) - mdates.date2num(begin_time),
                                   axes[i].get_ylim()[1] - axes[i].get_ylim()[0],
                                   alpha=0.4, color='green'))
                    
                    
            for begin_time,end_time in zip(antime,anend):
                begin_time=begin_time-timedelta(minutes=5)
                end_time=end_time+timedelta(minutes=5)
                for i,ax in enumerate(axes):
                    axes[i].add_patch(Rectangle((mdates.date2num(begin_time), axes[i].get_ylim()[0]),
                                   mdates.date2num(end_time) - mdates.date2num(begin_time),
                                   axes[i].get_ylim()[1] - axes[i].get_ylim()[0],
                                   alpha=0.4, color='red'))    
                    
        axes[2].xaxis.set_major_locator(mdates.AutoDateLocator())  # Set ticks every 6 hours
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[2].legend(loc="upper left")

        # Customize the overall plot
        plotTime=st[0].stats.starttime-timedelta(minutes=1)
        plotname=plotTime.strftime('%Y%m%d%H%M')
        plt.subplots_adjust(hspace=0, wspace=0)  # Adjust the right margin as needed
        plt.tight_layout()  # Increase padding between subplots
        fig.savefig(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{sensor}-{plotname}_{plottype}_test.png')
        plt.clf()
        plt.close(fig)
        gc.collect()  
    return


def getMaxMin(st,mt='s'):
    result=[]
    
    t=st[0].stats.starttime
    z=0
    lon=0
    lat=0
    
    
    if len(st)>1:
        try:
            if mt=='s':
                z=st.select(channel='*1')[0].data
                lon=st.select(channel='*2')[0].data
                lat=st.select(channel='*3')[0].data
            if mt=='d':
                z=st.select(channel='*Z')[0].data
                lat=st.select(channel='*E')[0].data
                lon=st.select(channel='*N')[0].data
        except:pass
        t=t
        if mt=='d':t=t+timedelta(minutes=1)
        if mt == 's':
            n=lon.size
            # temp=np.min(heapq.nlargest(n//40, lon))
            result.append([t.strftime('%Y/%m/%d'),
                           t.strftime('%H:%M:%S'),
                           st[0].stats.station,
                           np.max(z),
                           np.max(lon),np.max(lat),
                           np.min(z),np.min(lon)
                           ,np.min(lat),
                           st[0].stats.endtime.strftime('%Y/%m/%dT%H:%M:%S')])
        else:
            result.append([t.strftime('%Y/%m/%d'),
                           t.strftime('%H:%M:%S'),
                           st[0].stats.station,
                           np.max(z),
                           np.max(lon),np.max(lat),
                           np.min(z),np.min(lon)
                           ,np.min(lat),
                           st[0].stats.endtime.strftime('%Y/%m/%dT%H:%M:%S')])
    else:

        z=st.select(channel='B*Z')[0].data
        
        t=t
        result.append([t.strftime('%Y/%m/%d'),
                       t.strftime('%H:%M:%S'),
                       st[0].stats.station,
                       np.max(z),
                       0,0,
                       np.min(z),0
                       ,0,
                       st[0].stats.endtime.strftime('%Y/%m/%dT%H:%M:%S')])
    return result

        
def fixCorrupt(st):
    window = np.hanning(st[0].stats.npts)
    # flattop_window=scipy.signal.nuttall(st[0].stats.npts)
    for i,tr in enumerate (st):
        st[i].data=st[i].data * window 
        # st[i].data=st[i].data * flattop_window 
    return st
        
def trimStream(st,current,evttimes,antimes):
        
    for i,funct in enumerate(evttimes):
        st=st.cutout(funct[0],funct[1])
        
    # for i,ant in enumerate(antimes):
    #     st=st.cutout(ant[0],ant[1])    
    st=st.merge(method=0,fill_value=0)

    return st









mydir='/home/jeff/ffon/web/export/{}_{}_{}'.format(myyear,
            mymonth,
            myday,)

if not os.path.isdir(mydir):os.mkdir(mydir)








with Pool(processes=4) as pool:
    cn.stst=pool.map(readStream,zip(cn.stfiles,cn.stfiles2))

with Pool(processes=4) as pool:
    cn.stst=pool.map(changeTime,cn.stst)
    
# # for i,st in enumerate(cn.stst):
# #     cn.stst[i]=cn.stst[i].slice(UTCDateTime(f'{myyear}-{mymonth}-{myday}T14:30'),
# #                                     UTCDateTime(f'{myyear}-{mymonth}-{myday2}T23:26'))
# # for i,st in enumerate(cn.stst):cn.stst[i]=cn.stst[i].cutout(UTCDateTime(2024,9,14,19,29),UTCDateTime(2024,9,14,19,40))



print('Merging')

for i,st in enumerate(cn.stst):
    cn.stst[i]=cn.stst[i].select(sampling_rate=1000).merge(method=0,fill_value='latest')
    cn.stst[i]=cn.stst[i].sort()

# cn.stst[1]=Stream()
print('Calculating Stress')

cn.stst=cn.getStress()


# for i,st in enumerate(myst):plotSt2(myst[i])
print('Plotting Stress')

with Pool(processes=4) as pool:
    pool.map(plotStrain,cn.stst)



# cn.stst[1]+=cn.disst[1]

with Pool(processes=4) as pool:
    evtMax=pool.map(plotEvtst,cn.stst)

# pfs=[]
# i=0
# while i<len(evtMax):
#     try:
#         if not isinstance(evtMax[0][i][0],list):
#             pfs.append(evtMax[0].pop(i))
#             continue
#     except:break
#     i+=1
    
# raise ('error')   
        
print('Reading Displacement')
with Pool(processes=4) as pool:
    cn.disst=pool.map(readStream,zip(cn.vfiles,cn.vfiles2))

with Pool(processes=4) as pool:
    cn.disst=pool.map(changeTime,cn.disst)
    



cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2025,4,26,3,2),UTCDateTime(2025,4,26,3,7))
cn.disst[1]=cn.disst[1].cutout(UTCDateTime(2024,12,10,1,55),UTCDateTime(2024,12,10,1,56))
cn.disst[1]=cn.disst[1].cutout(UTCDateTime(2024,12,13,21,49,30),UTCDateTime(2024,12,13,21,49,40))

for i,st in enumerate(cn.disst):
    cn.disst[i]=cn.disst[i].cutout(UTCDateTime(2025,1,22,15,41),UTCDateTime(2025,1,22,15,53))
    cn.disst[i]=cn.disst[i].cutout(UTCDateTime(2025,1,22,16,18),UTCDateTime(2025,1,22,16,19))

# for i,tr in enumerate(cn.disst[0]):
#     if tr.stats.channel=='BSZ':
#         cn.disst[0][i]=temp[0]
# del temp
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2025,4,29,9,59),UTCDateTime(2025,4,29,10,1))
cn.disst[1]=cn.disst[1].cutout(UTCDateTime(2025,10,16,9,0),UTCDateTime(2025,10,16,9,10))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,1,6,0,40),UTCDateTime(2026,1,6,0,55))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,1,6,8,10),UTCDateTime(2026,1,6,8,40))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,1,7,18,0),UTCDateTime(2026,1,7,21,40))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,1,8,20,40),UTCDateTime(2026,1,8,23,0))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,1,8,9,40),UTCDateTime(2026,1,8,10,20))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,1,8,13,15),UTCDateTime(2026,1,8,13,30))
cn.disst[1]=cn.disst[1].cutout(UTCDateTime(2026,1,8,6,22,15),UTCDateTime(2026,1,8,6,22,45))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,1,31,8,30),UTCDateTime(2026,1,31,9,30))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,2,3,7,0),UTCDateTime(2026,2,3,7,30))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,2,3,8,30),UTCDateTime(2026,2,3,9,0))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,2,16,4,50),UTCDateTime(2026,2,16,5,4))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,5,9,16,40),UTCDateTime(2026,5,9,16,50))
cn.disst[0]=cn.disst[0].cutout(UTCDateTime(2026,5,10,15,0),UTCDateTime(2026,5,12,16,50))
cn.disst[1]=cn.disst[1].cutout(UTCDateTime(2026,5,10,15,0),UTCDateTime(2026,5,12,16,50))
cn.disst[1]=generalCutout(cn.disst[1])

# cn.disst[1]=cn.disst[1].cutout(UTCDateTime(2025,3,28,16,0,0),UTCDateTime(2025,3,28,23,59,59))

n=cn.disst[1].select(channel='*Z').copy()
n=specialCutout(n)
n=n.select(sampling_rate=1000).merge(method=0,fill_value=0)



for i,st in enumerate(cn.disst):cn.disst[i]=cn.disst[i].select(sampling_rate=1000).merge(method=0,fill_value=0)
print('Plotting')

for i,tr in enumerate(cn.disst[1]):cn.disst[1][i].data=cn.disst[1][i].data.astype(np.float32)

for i,tr in enumerate(cn.disst[0]):
    cn.disst[0][i].data=NewCN(cn.disst[0][i].data, np.zeros(8))[-1]
    
for i,tr in enumerate(cn.disst[1]):
    cn.disst[1][i].data=NewCN(cn.disst[1][i].data, np.zeros(8))[-1]

# for i,tr in enumerate(cn.disst[1]):
#     if tr.stats.channel=="BHZ" or tr.stats.channel=="BSZ":
#         cn.disst[1][i].data=n[0].data



nz=cn.disst[0].select(channel='*Z')
sz=cn.disst[1].select(channel='*Z')
nzp=obspy.Stream()
szp=obspy.Stream()
s_res=obspy.Stream()
n_res=obspy.Stream()
for i,v in enumerate(mydate):
    currt=mydate[i]-timedelta(minutes=2)
    s_strain=obspy.read(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{currt.strftime("%Y%m%d%H%M")}_s.mseed')
    
    f=execute_auto_calibration(s_strain,sz.slice(s_strain[0].stats.starttime,s_strain[0].stats.endtime))
    s_strain[0].data=s_strain[0].data*2.0*np.abs(f)
    s_res+=s_strain
    
    # n_strain=obspy.read(f'/home/jeff/ffon/web/export/{myyear}_{mymonth}_{myday}/{currt.strftime("%Y%m%d%H%M")}_n.mseed')
    # fn=execute_auto_calibration(n_strain,nz.slice(n_strain[0].stats.starttime,n_strain[0].stats.endtime))
    # n_strain[0].data=n_strain[0].data*2.0*np.abs(fn)
    # n_res+=n_strain    
    # nzp+=apply_shape_transfer(nz.slice(mydate[i],evtend[i]),n_strain,force_p2p=z_diff[i][0])
    # nz=nz.cutout(mydate[i],evtend[i])
    # nz+=nzp
    # szp+=apply_shape_transfer(sz.slice(mydate[i],evtend[i]),s_strain,force_p2p=z_diff[i][1])
    # sz=sz.cutout(mydate[i],evtend[i])
    # sz+=szp

for i,tr in enumerate(sz):sz[i].data=sz[i].data.astype(np.float32)
sz=sz.merge(method=0,fill_value=0)

for i,v in enumerate(cn.disst[1]):
    # if cn.disst[0][i].stats.channel=='BHZ':cn.disst[0][i]=nz[0]
    if cn.disst[1][i].stats.channel=='BHZ':cn.disst[1][i]=sz[0]
# nz=nz.merge(method=0,fill_value=0)
plotEvt(s_res)

for i,st in enumerate(cn.disst):cn.disst[i]=cn.disst[i].merge(method=0,fill_value=0)

# cn.plotDay()

# with Pool(processes=2) as pool:
#     pool.map(plotEvt,cn.disst)
    

    
















