#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:57:12 2022

@author: jeff
"""

import os
import re
import weasyprint
from jinja2 import Environment, FileSystemLoader
from datetime import datetime, timedelta
import glob
from obspy import UTCDateTime,Stream
import pandas as pd
import plotly.graph_objects as go
import plotly.io as io
from plotly.offline import plot
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product,combinations
from bs4 import BeautifulSoup
from weasyprint import HTML,CSS
from functools import reduce
import time


dfstrain=[]
gaugeLoc=['N_GDR_BF_INS(Max)','S_GDR_TF(Max)','S_GDR_BF_OUT(Max)',
          'S_GDR_BF_OUT(Min)','N_GDR_BF_INS(Min)','S_GDR_TF(Min)',
 'N_GDR_BF_INS(Max)','N_GDR_BF_OUT(Max)','N_GDR_TF(Max)',
 'N_GDR_BF_INS(Min)','N_GDR_BF_OUT(Min)','N_GDR_TF(Min)']

current=datetime.now()-timedelta(days=1)
# current=datetime(2026,2,2,0,0,0)
print(current)

templates_dir='/home/jeff/ffon/web'
myCSS='/home/jeff/ffon/web/report.css'
DEST_DIR = f'./web/export/{current.year}_{current.month}_{current.day}/\
{current.year}_{current.month}_{current.day}.html'
env = Environment( loader = FileSystemLoader(templates_dir) )

template = env.get_template('report.html')


outputfilename=f'{current.year}_{current.month}_{current.day}.pdf'
df82=pd.DataFrame(columns=['Start Date (CST)','Start Time (CST)','Box ID','N_GDR_BF_INS(Max)',
    'N_GDR_BF_OUT(Max)','N_GDR_TF(Max)','N_GDR_BF_INS(Min)','N_GDR_BF_OUT(Min)','N_GDR_TF(Min)','End Time (CST)'
])

df80=pd.DataFrame(columns=['Start Date (CST)','Start Time (CST)','Box ID','S_GDR_BF_OUT(Max)','S_GDR_BF_INS(Max)','S_GDR_TF(Max)',
              'S_GDR_BF_INS(Min)','S_GDR_BF_OUT(Min)','S_GDR_TF(Min)','End Time (CST)'])
    
df82=pd.DataFrame(columns=['Start Date (CST)','Start Time (CST)','Box ID','N_GDR_BF_OUT(Max)','N_GDR_BF_INS(Max)','N_GDR_TF(Max)',
          'N_GDR_BF_INS(Min)','N_GDR_BF_OUT(Min)','N_GDR_TF(Min)','End Time (CST)'])
if os.path.exists(f'/home/jeff/ffon/strain/082_{current.year}-{str(current.month).zfill(2)}'+
                 f'-{str(current.day).zfill(2)}_logs.csv'):
    df82=pd.read_csv(f'/home/jeff/ffon/strain/082_{current.year}-{str(current.month).zfill(2)}'+
                     f'-{str(current.day).zfill(2)}_logs.csv',sep=',',index_col=False)
    df82['Start Time (CST)'] = df82['Start Time (CST)'].str[0:5]
    df82=df82[['Start Date (CST)','Start Time (CST)','Box ID','N_GDR_BF_OUT(Max)','N_GDR_BF_INS(Max)','N_GDR_TF(Max)',
              'N_GDR_BF_INS(Min)','N_GDR_BF_OUT(Min)','N_GDR_TF(Min)','End Time (CST)']]
if os.path.exists(f'/home/jeff/ffon/strain/080_{current.year}-{str(current.month).zfill(2)}'+
                 f'-{str(current.day).zfill(2)}_logs.csv'):
    df=pd.read_csv('/home/jeff/ffon/S5_{}-{}-{}_logs.csv'.format(current.year,
                str(current.month).zfill(2),str(current.day).zfill(2)),sep=',',index_col=False)
    df['Start Time (CST)'] = df['Start Time (CST)'].str[0:5]
    
    df80=pd.DataFrame()

    df80=pd.read_csv(f'/home/jeff/ffon/strain/080_{current.year}-{str(current.month).zfill(2)}'+
                 f'-{str(current.day).zfill(2)}_logs.csv',sep=',',index_col=False)
    
    df80['Start Time (CST)'] = df80['Start Time (CST)'].str[0:5]
    df80=df80[['Start Date (CST)','Start Time (CST)','Box ID','S_GDR_BF_OUT(Max)','S_GDR_BF_INS(Max)','S_GDR_TF(Max)',
                  'S_GDR_BF_INS(Min)','S_GDR_BF_OUT(Min)','S_GDR_TF(Min)','End Time (CST)']]
    
dfstrain=pd.concat([df80,df82])

dfstrain=dfstrain.drop('End Time (CST)',axis=1)
dfstrain=dfstrain.drop('Box ID',axis=1)
dfstrain=dfstrain.fillna(0)
dfstrain=dfstrain.set_index(['Start Date (CST)','Start Time (CST)'])
dfstrain=dfstrain.groupby(dfstrain.index).sum()

maxCol=lambda x: max(x.min(), x.max(), key=abs)


def sortTime(a):
    a=re.findall(r'\d{10,16}', a)[0]
    st=int(a)
    return st

def getTime(a):
    a=sortTime(a)
    st=UTCDateTime(str(a))
    st=st.strftime('%H:%M')
    return st




def makeGraph(b,travel,tinfo,rinfo,maxinfo,d='displacement'):

    coCode={'NBTM-Z':'#055c9d',
            'SBTM-Z':'#219eff',
            'NBTM-Longitudinal':'#3d550c',
            'SBTM-Longitudinal':'#b1d8b7',
            'NBTM-Lateral':'#6f0000',
            'SBTM-Lateral':'#ecd5d0',
            'N_GDR_BF_OUT':'#055c9d',
            'N_GDR_BF_INS':'#219eff',
            'S_GDR_BF_INS':'#3d550c',
            'S_GDR_BF_OUT':'#b1d8b7',
            'N_GDR_TF':'#6f0000',
            'S_GDR_TF':'#ecd5d0',}

    fig=go.Figure()
    for i,v in enumerate(travel[0]):
        fig.add_trace(
            go.Bar(
            x=tinfo,
            y=[myval[i] for myval in travel],
            base=[mytravel[i] for mytravel in b],
            name=rinfo[i],
            meta=[rinfo[i]],
            text=['{0:0.3f}'.format(mytravel[i]) for mytravel in travel],
            textposition = "none", 
            
            marker_color=coCode[rinfo[i]],
            # orientation='h',
            hovertemplate=
            "<b>%{meta}</b><br><br>" +
            "Time: %{x}<br>" +
            "Minimum Displacement(mm): %{y}<br>" +
            "Maximum Displacement(mm) %{base}<br>"+
            "Total Dynamic Displacement(mm) %{text}"
            "<extra></extra>"
            ),
            )
    fig.update_xaxes(
            type='category',
            title="Event Time (CST)",
            tickmode = 'array',
            tickvals = tinfo,
            ticktext = [a.split('T')[-1] for a in tinfo],


            # dtick=86400000
            )
    if d=='displacement':
        fig.update_layout(
        margin={
            'l':0, #left margin
            'r':0, #right margin
            'b':0, #bottom margin
            't':0, #top margin
        },
        legend=dict(
        orientation="h",
    
        ),
    
        yaxis_title="Dynamic Displacement(mm)",
        legend_title="Channels",
    
            
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="RebeccaPurple"
        ) )   
    else: 
        fig.update_layout(
        margin={
            'l':0, #left margin
            'r':0, #right margin
            'b':0, #bottom margin
            't':0, #top margin
        },
        legend=dict(
        orientation="h",
    
        ),
    
        yaxis_title="Stress(psi)",
        legend_title="Channels",
    
            
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="RebeccaPurple"
        ) )   
    # plot(fig)
    # fig.write_html("Sep19.html")
    return fig

def makeSummary(df):
    coCode={'North Bottom(Z)':'#219eff','N_GDR_BF_INS':'#219eff',
            'South Bottom(Z)':'#055c9d','N_GDR_BF_OUT':'#055c9d',
            'North Bottom(longitudinal)':'#b1d8b7','N_GDR_TF':'#b1d8b7',
           'South Bottom(longitudinal)':'#3d550c', 'S_GDR_BF_OUT':'#3d550c',
            'North Bottom(Lateral)':'#ecd5d0','N_GDR_BF_INS':'#ecd5d0',
            'South Bottom(Lateral)':'#6f0000','STS-GDR-TF':'#6f0000'}
    
    lrank={'North Bottom(Z)':9,'Left Pedestal(Z)':3,
            'South Bottom(Z)':12,'Right Pedestal(Z)':6,
            'North Bottom(longitudinal)':8,'Left Pedestal(longitudinal)':2,
           'South Bottom(longitudinal)':11, 'Right Pedestal(longitudinal)':5,
            'North Bottom(Lateral)':7,'Left Pedestal(Lateral)':1,
            'South Bottom(Lateral)':10,'Right Pedestal(Lateral)':4}
    # print(df.max().to_frame().T.values)
    maxInd,maxVal=[df.max().to_frame().T.columns.values[1:],
                   df.max().to_frame().T.values[0][1:]]

    fig=go.Figure()

    df=df.tail(14)
    for i,v in enumerate(reversed(df.columns.values[1:])):
        fig.add_trace(
            go.Bar(
            y=df['Date'],
            x=df[v],
            name=v,
            textposition = "none", 
            marker_color=coCode[v],
            meta=v,
            orientation='h',
            legendrank=lrank[v],
            hovertemplate=
            "<b>%{meta}</b><br><br>" +
            "Time: %{x}<br>" +
            "Daily Dynamic Displacement(mm): %{y}<br>" +
            "<extra></extra>"
            ),
            )
    fig.update_layout(

    margin={
        'l':0, #left margin
        'r':0, #right margin
        'b':0, #bottom margin
        't':0, #top margin
    },
    
    legend=dict(
    orientation="h",
    traceorder='reversed',
    uirevision=3,
    font=dict(
            family="Courier",
            size=12,
            color="black"
        )
    
    ),
    yaxis_title="Date (CST 00:00-23:59)",
    xaxis_title="Daily Maximum Total Dynamic Displacement (Max-Min) in mm",
    legend_title="Channels",

        
    font=dict(
        family="Courier New, monospace",
        size=14,
        color="RebeccaPurple"
    ) ) 
    
    fig.update_yaxes(

    dtick=86400000
    )
    if len(maxInd) >6:
        for i,v in enumerate(maxInd):
            print(v,maxVal[i])
            if i%2==1:
                print(maxVal[i],maxVal[i-1])
                fig.add_vrect(x0=maxVal[i],x1=maxVal[i-1], 
                              line_width=0, fillcolor=coCode[v],opacity=0.25)
    else:
        for i,v in enumerate(maxInd):
            fig.add_vline(x=maxVal[i], line_width=3, line_dash="dash", line_color=coCode[v])
    # plot(fig)
    # fig.write_html("Sep19.html")
    return fig



def plotSt(st):
    st.plot(outfile='{}.png'.format(
        st[0].stats.station),
        show=False,)




def writeDf(current=current):
    print('Reading DF')
    desdict={"NBTM":"North",
             "SBTM":"South",
   
    }

    df=pd.read_csv('/home/jeff/ffon/S5_{}-{}-{}_logs.csv'.format(current.year,
                str(current.month).zfill(2),str(current.day).zfill(2)),sep=',',index_col=False)
    df['Start Time (CST)'] = df['Start Time (CST)'].str[0:5]
    maxCol=lambda x: max(x.min(), x.max(), key=abs)
    disinfo=[desdict[des] for des in df['Box ID'].values]
    pt=df.pivot_table(index=['Start Date (CST)','Start Time (CST)'], 
        columns='Box ID', values=['Z(+ve)', 'Z(-ve)','Longitudinal(+ve)','Longitudinal(-ve)',
                    'Lateral(+ve)','Lateral(-ve)'])
    basicElements =['Z(+ve)',
                    'Z(-ve)', 
                    'Longitudinal(+ve)', 
                    'Longitudinal(-ve)',
                    'Lateral(+ve)', 
                    'Lateral(-ve)']
    
    simpleElements =['Z',
                    'Longitudinal', 
                    'Lateral']
    
    maxElements =['Z MAX',
                    'Longitudinal MAX', 
                    'Lateral MAX', ]
    
    boxes=['NBTM','SBTM']
    boxes2=['NBTM','SBTM']

    df['Box ID'] = pd.Categorical(df['Box ID'], ['NBTM','SBTM'])
    pt=df.pivot_table(index=['Start Date (CST)','Start Time (CST)'],              
    columns=['Box ID',], values=['Z(+ve)', 'Z(-ve)','Longitudinal(+ve)','Longitudinal(-ve)',
            'Lateral(+ve)','Lateral(-ve)'])
    pt=pt[pt.notna().all(1)]
    

    pt0=pt.stack()
    pt0=pt0[basicElements]
    
    dinfo=[desdict[des[-1]] for des in pt0.index.values]
    pt0=pt0.reset_index()
    
    pt0['Description'] = dinfo
    pt0=pt0.set_index(['Start Date (CST)', 'Start Time (CST)',
                      'Description', 'Box ID'])
    
    # pt0.to_csv('{}.csv'.format(current.strftime('%Y-%m-%d')))
    


    channelTab=[]
    channelSim=[]
    maxTab=[]
    myMax=[]
    cnMax=[]
    topOnly=[]
    for b,e in product(boxes,basicElements):channelTab.append('{}-{}'.format(b,e))
    for b,e in product(boxes,simpleElements):channelSim.append('{}-{}'.format(b,e))
    for asd,ma in product(boxes,maxElements):maxTab.append('{}-{}'.format(asd,ma))
    for asd,ma in product(maxElements,boxes):myMax.append('{}-{}'.format(ma,asd))
    for asd,ma in product(boxes2,maxElements):cnMax.append('{}-{}'.format(asd,ma))

    pt.columns = [f'{j}-{i}' if j != '' else f'{i}' for i,j in pt.columns]
    needtoinsert=set(channelTab)-set(pt.columns)
    # Create a DataFrame with zero values for the new columns
    new_data = {col: [0] * len(pt) for col in needtoinsert}
    new_df = pd.DataFrame(new_data, index=pt.index)

    # Concatenate the original DataFrame and the new DataFrame along axis=1 (columns)
    pt = pd.concat([pt, new_df], axis=1)
    pt=pt[channelTab]
    ptg=pt[channelTab]
    # ptg= pt[(pt.max(axis=1) > 0.6) | (pt.min(axis=1) < -0.6)]
    # if ptg.shape[0] > 4: ptg= ptg[(pt.max(axis=1) > 1) | (pt.min(axis=1) < -1)]
    
    maxDict={}
    for box in boxes2:
        print(box)
        temp_df=df[df['Box ID']==box]
        if not temp_df.empty:
            maxDict[f'{box}-Z MAX']=(temp_df['Z(+ve)'] - temp_df['Z(-ve)']).max()
            maxDict[f'{box}-Longitudinal MAX']=(temp_df['Longitudinal(+ve)'] - temp_df['Longitudinal(-ve)']).max()
            maxDict[f'{box}-Lateral MAX']=(temp_df['Lateral(+ve)'] - temp_df['Lateral(-ve)']).max()

    

    tmp=pd.DataFrame([maxDict])
    # tmp.columns=maxTab
    for i, v in enumerate(myMax):
        try:
            tmp.insert(0,myMax[i],0,allow_duplicates=False)
        except:
            continue
    mytmp=tmp[myMax]

    mytmp.insert(0,'Start Date (CST)','{}-{}-{}'.format(current.year
                                    ,str(current.month).zfill(2)
                                    ,str(current.day).zfill(2)))
    cntmp=tmp[cnMax]
    cntmp.insert(0,'Start Date (CST)','{}-{}-{}'.format(current.year
                                    ,str(current.month).zfill(2)
                                    ,str(current.day).zfill(2)))
    

    
    return [ptg,pt0,mytmp,cntmp]

def lineCNtable(df_html):
    soup = BeautifulSoup(df_html, 'html.parser')
    date_cols = soup.find_all(['tr','th'])
    tgt=''
    
    for i,stuff in enumerate(date_cols):
        
        check=stuff.find('th',string='Date')
    
    
        if i==0: fr=stuff
        if check :
            tgt=stuff
    
    
    tgt.decompose()
    date_tag = soup.new_tag("th", rowspan=2,halign="left")
    date_tag.string='Date'
    fr.insert(3,date_tag)
    fr.find('th').decompose()
    
    for x in soup.find_all('th'):
        if len(x.get_text()) == 0:
          x.decompose() 
    return soup.decode_contents()

def lineColumns(df_html,pt0):
    soup = BeautifulSoup(df_html, 'html.parser')
    empty_cols = soup.find('thead').find_all(lambda tag: not tag.contents)


    for tag, col in zip(empty_cols,list(pt0.index.names)+pt0.columns.values.tolist()):
        tag.string = col
        
    temp=soup.find_all('tr')
    temp[1].decompose()
    pt0_html=soup.decode_contents()
    return pt0_html



def colorPT(df_html):
    soup = BeautifulSoup(df_html, 'html.parser')
    ind_rows = soup.find_all(['tr'])

    for i,tag in enumerate(ind_rows):
        if i==0: continue
        ele=tag.find_all(['th','td'])
        for j, indtag in enumerate(ele):
            if len(ele)>11:
                if j<=0: continue
            elif len(ele)>9:
                if j<=1:continue
            elif len(ele)>8:
                if j==0: continue
            if i%2==1:indtag['class']='blue' 
            if i%2==0:indtag['class']='green'
                
    pt0_html=soup.decode_contents()
    return pt0_html

def colorCN(df_html):
    soup=BeautifulSoup(df_html, 'html.parser')
    ro=soup.find_all('tr')

    for i,v in enumerate(ro):
        if i==0: 
            mytgt=v.find_all('th')
            for j,ele in enumerate(mytgt):
                if j%2==0 and j:ele['class']='green'
                if j%2==1:ele['class']='blue'
            continue
        elif(i==1): mytgt=v.find_all('th')
        else:mytgt=v.find_all('td')
        for j,ele in enumerate(mytgt):
            if j-3<0:
                ele['class']='blue'
                continue
            if j-6<0:
                ele['class']='green'
                continue

    my_html=soup.decode_contents()
    return my_html


def PlotlyStrain(pt):
    
    if pt.empty:return

    channelSim=['N_GDR_BF_OUT','N_GDR_BF_INS','N_GDR_TF','S_GDR_BF_INS','S_GDR_BF_OUT','S_GDR_TF']
    
    sub=['(Max)','(Min)']
    a=[]
    for b,e in product(channelSim,sub):a.append('{}{}'.format(b,e))
    pt=pt[a]
    b=pt.values
    b=b.reshape(len(b),6,-1)
    
    startVal=np.min(b,axis=2)
    maxVal=np.max(b,axis=2)
    print(b)
    b=b.astype(float)
    travel=-1*np.diff(b,axis=2).reshape(len(b),-1)
    ti=pt.index.values
    myt=['{}T{}'.format(d,t) for d,t in ti]
    
    asd=maxVal-startVal
    mask = np.abs(startVal)>np.abs(maxVal)
    asd=np.where(mask,asd*-1,asd)
    

    index = pd.MultiIndex.from_tuples(ti)
    temp_pt=pd.DataFrame(data=asd,columns=channelSim,index=index)
    
    # if len(pt)>4:
    #     temp_pt= temp_pt[(temp_pt.abs() > 1e-04).any(axis=1)]
    #     pt=pt.loc[temp_pt.index]
    #     b=pt.values
    #     b=b.reshape(len(b),6,-1)
    #     startVal=np.min(b,axis=2)
    #     maxVal=np.max(b,axis=2)
    #     travel=-1*np.diff(b,axis=2).reshape(len(b),-1)
    #     ti=pt.index.values
    #     myt=['{}T{}'.format(d,t) for d,t in ti]


    return [startVal,travel,myt,channelSim,maxVal,temp_pt]

    
def setupPlotly(pt):
    if pt.empty:return
    
    simpleElements =['Z',
                    'Longitudinal', 
                    'Lateral']
    boxes=['NBTM','SBTM']
    # boxes=['SBTM']
    channelSim=[]
    for b,e in product(boxes,simpleElements):channelSim.append('{}-{}'.format(b,e))
    if len(pt)>4: pt=pt[pt['SBTM-Z(+ve)']>0.4]
    b=pt.values
    b=b.reshape(len(b),6,-1)
    # b=b.reshape(len(b),3,-1)
    startVal=np.min(b,axis=2)
    maxVal=np.max(b,axis=2)
    travel=-1*np.diff(b,axis=2).reshape(len(b),-1)
    ti=pt.index.values

    myt=['{}T{}'.format(d,t) for d,t in ti]

    return [startVal,travel,myt,channelSim,maxVal]

def getLowfreq(ar):
    i=0
    result=[]
    while i<len(ar):
        if 'train-enhanced' in ar[i]: 
            temp=ar.pop(i)
            result.append(temp)
        else:i+=1
    return result
    
    
NBTM=sorted(glob.glob('/home/jeff/ffon/web/export/{}/NBTM*.png'.format(current.strftime("%Y_%m_%d"))),key=sortTime)
SBTM=sorted(glob.glob('/home/jeff/ffon/web/export/{}/SBTM*.png'.format(current.strftime("%Y_%m_%d"))),key=sortTime)
s080=sorted(glob.glob('/home/jeff/ffon/web/export/{}/080*.png'.format(current.strftime("%Y_%m_%d"))),key=sortTime)
s082=sorted(glob.glob('/home/jeff/ffon/web/export/{}/082*.png'.format(current.strftime("%Y_%m_%d"))),key=sortTime)

dayplot=[NBTM.pop(0),SBTM.pop(0),s082.pop(0),s080.pop(0)]
# dayplot=[NBTM.pop(0),SBTM.pop(0),'',s080.pop(0)]
# dayplot=[NBTM.pop(0),SBTM.pop(0),s082.pop(0),'']
# dayplot=['',SBTM.pop(0),s082.pop(0),'']
# dayplot=['',SBTM.pop(0),'','']
pic_meta=[]




fileLocations=[]
[ptg,pt0,mytmp,cntmp]=writeDf()



mytmp=mytmp.replace(0,'N/A')
cntmp=cntmp.replace(0,'N/A')





mytime=[getTime(ti) for ti in s080]

fig=None
stfig=None
maxCol=lambda x: max(x.min(), x.max(), key=abs)
maxRow=lambda x: max(x.min(), x.max(), key=abs)
if not ptg.empty:
    startVal,travel,myt,channelSim,maxVal=setupPlotly(ptg)
    fig=makeGraph(startVal,travel,myt,channelSim,maxVal,d='displacement')
    
if not dfstrain.empty:
    stVal,st_travel,st_t,st_channel,st_maxVal,stDaily=PlotlyStrain(dfstrain)
#     stfig=makeGraph(stVal,st_travel,st_t,st_channel,st_maxVal,d='strain')
    stDaily = stDaily.fillna(0)
    result=stDaily.apply(maxCol,axis=0)
    t2=stDaily.reset_index(names=['Start Date (CST)','Start Time (CST)'])
    t2=t2[['Start Date (CST)', 'Start Time (CST)', 'N_GDR_BF_OUT', 'N_GDR_BF_INS',
           'N_GDR_TF', 'S_GDR_BF_OUT','S_GDR_BF_INS', 'S_GDR_TF']]
    new_element = pd.Series([f'{current.strftime("%Y-%m-%d")}'])
    result = pd.concat([new_element, result], ignore_index=True)
    result = pd.DataFrame([result.values], columns=result.index,)





mytmp.to_csv('ON_Maximum.csv', mode='a', header=False,index=False) 
cntmp.to_csv('ON_daily_table.csv', mode='a', header=False,index=False)

if not dfstrain.empty:
    col5_copy = result.iloc[:, 4].copy()

    # Assign column 6 to column 5's position
    result.iloc[:, 4] = result.iloc[:, 5]
    
    # Assign the copied column 5 to column 6's position
    result.iloc[:, 5] = col5_copy
    result.to_csv('ON_Strain.csv', mode='a', header=False,index=False)


if not dfstrain.empty:t2.to_csv(f'/home/jeff/ffon/web/export/{current.strftime("%Y_%m_%d")}/ON_Table2.csv', mode='w',
          header=True,index=False)
time.sleep(3)

topvalues=['Date', 'North Bottom(Z)',
           
           
       'South Bottom(Z)',
       'North Bottom(longitudinal)',
       'South Bottom(longitudinal)',
       'North Bottom(Lateral)',
       'South Bottom(Lateral)']



mydaily=pd.read_csv('ON_Maximum.csv')
cndaily=pd.read_csv('ON_daily_table.csv',header=[0,1],index_col=0)
straindaily=pd.read_csv('ON_Strain.csv',header=[0,1],index_col=0)


mydaily=mydaily.drop_duplicates()
cndaily=cndaily.drop_duplicates()
straindaily=straindaily.drop_duplicates()
cndaily=cndaily.tail(93)
straindaily=straindaily.tail(93)

topdaily=mydaily[topvalues]


cndaily = cndaily.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
temp=[]
sta=''
newCol=[]

for i,v in enumerate(cndaily.columns):
    temp=list(v)
    if v[0]:
        sta=v[0]
    else:temp[0]=sta 
    newCol.append(temp)




cndaily.columns=pd.MultiIndex.from_tuples(newCol)
cndaily.index.name='Date'
dailyPic=makeSummary(mydaily)


if fig:
    fig.write_image(f"/home/jeff/ffon/web/export/{current.strftime('%Y_%m_%d')}/today.png",
                    scale=1,width=800, height=950)
# if stfig:
#     stfig.write_image(f"/home/jeff/ffon/web/export/{current.strftime('%Y_%m_%d')}/strain_today.png",
#                     scale=1,width=800, height=950)
dailyPic.write_image(f"/home/jeff/ffon/web/export/{current.strftime('%Y_%m_%d')}/Overall_summary.png",
                scale=1,width=1000, height=1150)



straindaily = straindaily.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
temp=[]
sta=''
newCol=[]

for i,v in enumerate(straindaily.columns):
    temp=list(v)
    if v[0]:
        sta=v[0]
    else:temp[0]=sta 
    newCol.append(temp)
    
    
straindaily.columns=pd.MultiIndex.from_tuples(newCol)
straindaily.index.name='Date'
cndaily=cndaily.round(3)
pt0=pt0.replace(0,'N/A')
cndaily=cndaily.fillna('N/A')
cndaily=cndaily.replace(0,'N/A')






pt0_html=pt0.round(3).to_html(border=1,col_space=1,header='False')   


if not dfstrain.empty:
    strainTable=pd.read_csv(f'/home/jeff/ffon/web/export/{current.strftime("%Y_%m_%d")}/ON_Table2.csv')
    strainTable = strainTable.loc[(strainTable!=0).any(axis=1)]
    strainTable=strainTable.round(1)
    
cndaily_html=cndaily.to_html(border=1,col_space=1,table_id='daily')


straindaily=straindaily.round(1)
straindaily=straindaily.replace(0,'N/A')

# strainTable=strainTable.replace(0,'N/A')


cndaily_html=lineCNtable(cndaily_html) 
cndaily_html=colorCN(cndaily_html)

straindaily=straindaily.round(1)
straindaily=straindaily.sort_values(by=['Date'])
straindaily_html=straindaily.round(1).to_html(border=1,col_space=1,table_id='daily')
straindaily_html=lineCNtable(straindaily_html) 
straindaily_html=colorCN(straindaily_html)

pt0_html=lineColumns(pt0_html,pt0)
pt0_html=colorPT(pt0_html)
t2_html="No Event Found."
if not dfstrain.empty:
    t2_html=strainTable.round(1).to_html(border=1,col_space=1,index=False) 
    #t2_html=lineColumns(t2_html,strainTable)
    t2_html=colorPT(t2_html)




filename = os.path.join(f'./web/export/{current.year}_{current.month:02}_{current.day:02}/\
{current.year}_{current.month:02}_{current.day:02}.html')

# s082=[]

if len(s082)>len(s080) and (any(s082) or any(s082)):
    s080=[] 
    NBTM=[]
    SBTM=[]
    for i,v in enumerate(s082):
        s82=v.replace('/082-','/080-')
        if os.path.isfile(v):s080.append(s82)
        else:s080.append('')
        sbtm=v.replace('/082-','/SBTM-')
        sbtm=sbtm.replace('_train','')
        sbtm=sbtm.replace('_other','')
        print('Doing SBTM')
        if os.path.isfile(sbtm):SBTM.append(sbtm)
        else:SBTM.append('')
        nbtm=v.replace('/082-','/NBTM-')
        nbtm=nbtm.replace('_train','')
        nbtm=nbtm.replace('_other','')
        if os.path.isfile(nbtm): NBTM.append(nbtm)
        else:NBTM.append('')
        
    for i,v in enumerate(s082):
    # for i,v in enumerate(s080):
        check=v.split('.')[0]
        check=check.split('_')[-1].capitalize()
        pic_meta.append(check)
            
elif(any(s082) or any(s082)):
    s082=[] 
    NBTM=[]
    SBTM=[]
    for i,v in enumerate(s080):
        s82=v.replace('/080-','/082-')
        if os.path.isfile(v):s082.append(s82)
        else:s082.append('')
        sbtm=v.replace('/080-','/SBTM-')
        sbtm=sbtm.replace('_train','')
        sbtm=sbtm.replace('_other','')
        print('Doing SBTM')
        if os.path.isfile(sbtm):SBTM.append(sbtm)
        else:SBTM.append('')
        nbtm=v.replace('/080-','/NBTM-')
        nbtm=nbtm.replace('_train','')
        nbtm=nbtm.replace('_other','')
        if os.path.isfile(nbtm): NBTM.append(nbtm)
        else:NBTM.append('')
    for i,v in enumerate(s080):
    # for i,v in enumerate(s080):
        check=v.split('.')[0]
        check=check.split('_')[-1].capitalize()
        pic_meta.append(check)     
else:
    for i,sbtm in enumerate(SBTM):
        pic_meta.append('Train')   
        # name, ext = sbtm.rsplit(".", 1)
        # new_filename = f"{name}_train.{ext}"
        # SBTM[i]=new_filename
        
    
    

        
# with open(filename, 'w') as fh:

base_url = os.path.dirname(os.path.realpath(__file__))

html_string=template.render(
    mydate=f'{current.year}-{current.month}-{current.day}',
    g1=f"/home/jeff/ffon/web/export/{current.strftime('%Y_%m_%d')}/today.png",
    g2=f"/home/jeff/ffon/web/export/{current.strftime('%Y_%m_%d')}/strain_today.png",
    ptst=t2_html,
    pt=pt0_html,
    cnmax=cndaily_html,
    strainmax=straindaily_html,
    wholeDay=dayplot,
    bottomNorth=NBTM,
    bottomSouth=SBTM,
    stSouth=s080,
    stNorth=s082,
    myInfo=pic_meta,
    myTime=mytime,
    gTemp=f"/home/jeff/ffon/web/export/{current.strftime('%Y_%m_%d')}/{current.strftime('%Y-%m-%d')}_temperature.png",
    # special=f"/home/jeff/ffon/web/export/{current.strftime('%Y_%m_%d')}/special-{current.strftime('%Y%m%d')}.png",

)

css = CSS(myCSS, base_url=base_url)
HTML(string=html_string,base_url=base_url).write_pdf('{}_report CN Mile 84.00 Fort Frances subdivision.pdf'.format(current.strftime('%Y-%m-%d'))
        ,stylesheets=[css],
            presentational_hints=True)
    
    
    

    
    
    
    
    