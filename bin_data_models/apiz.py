
import requests , json 
import contextlib
from bs4 import BeautifulSoup as bs 
import pandas as pd 

from termcolor import colored 

import traceback 

# import sys
# sys.path.append("../../../shared") 
# import zlogger

def getPage(url):
  try:
    with contextlib.closing( requests.get(url=url, stream=True) ) as rqst:
      return rqst.content 
  except requests.exceptions.RequestException as e:
    print( colored( f"Exception: {e}", 'red' ) )
    return None


def genPdTableFromHtml(htmld):
    print( colored('starting to parse df from table.html', 'blue') )
    dset_all = pd.read_html(  str(htmld) )
    dset_all = dset_all[0]   

    colz = ['Country,Other', 'TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths', 'TotalRecovered', 'ActiveCases', 
        'Serious,Critical', 'TotalTests']

    countriez = ['Kenya', 'World']
    
    df = dset_all[ dset_all['Country,Other'].isin( countriez) ][colz] 

    print( colored( "Created df from html : {}".format(df.shape),'red') )

    return df

def getStatsFromWorldometer(country='Kenya'):    
    KE_DATA = {}
    GLOBAL_DATA = {}

    url = "https://www.worldometers.info/coronavirus/" 
    today_tbl = '#main_table_countries_today'
    yester_tbl = '#main_table_countries_yesterday'

    html = bs( getPage(url), 'html.parser')  

    yester_df = genPdTableFromHtml( html.select( yester_tbl ) )     
    today_df = genPdTableFromHtml( html.select( today_tbl ) ) 

    print( f"Worldometer FOUND: yester: {yester_df.shape} and today: {today_df.shape }")
    ## reset col names to avoid changing ui w/r/t publicAPI naming
      
    colz = ['Country', 'TotalConfirmed', 'NewConfirmed', 'TotalDeaths', 'NewDeaths', 
            'TotalRecovered', 'ActiveCases', 'Serious,Critical', 'TotalTests']
        
    yester_df.columns = colz
    today_df.columns = colz

    ## TODO: make DRY
    
    def pullIntoDict(df_today, df_yester, places_list=['Kenya', 'World'], colz=colz):
        print(colored( f"Dictify: {places_list}", 'red'))
        RESULTS = []

        for place in places_list:
            ddict = {}
        
            dftmp = df_today[ df_today.Country == place ]
            dftmp_yester = df_yester[ df_yester.Country == place ]

            idx_today = dftmp.index[0]
            idx_yester = dftmp_yester.index[0]

            ddict['Country'] = place
        
            print(colored( f"Dictify: {ddict}", 'blue'))
            
            for col in colz:
                if col.startswith('New'):
                    print( colored(f'COl. Statswith.New: {col}', 'blue'))
                    val = dftmp.at[idx_today, col ] 
                    print( colored(f'COl. Statswith.New: {val}', 'blue'))
                    if val and (str(val).startswith("+") or str(val).startswith("-") ):
                        ddict[ f"{col}_direction"] = val[0]
                        ddict[ f"{col}"] = int(str(val[1:]).replace(",", ""))
                    elif str(val).startswith('nan'):
                        ddict[ f"{col}"] = 0
                    else:
                        ddict[ f"{col}"] = int(str(val).replace(",", ""))

                else:
                    ddict[ f"{col}"] = dftmp.at[idx_today, col ]         
            
            
            print(colored( f"Dictify: {ddict}", 'blue'))

            ddict['NewRecovered'] = dftmp.at[idx_today, 'TotalRecovered'] - dftmp_yester.at[idx_yester, 'TotalRecovered']

            print(colored( f"Dictify: {ddict}", 'red'))

            RESULTS.append( ddict  )
        return RESULTS[0], RESULTS[1]

    KE_DATA , GLOBAL_DATA = pullIntoDict(today_df, yester_df)

    # zlogger.log('stats.worldometer', f"KE: {KE_DATA}\nGLB: {GLOBAL_DATA}")
    return KE_DATA, GLOBAL_DATA


def getStatsFromPublicAPI(country='Kenya'):
    KE_DATA = None
    GLOBAL_DATA = None

    api_url = "https://api.covid19api.com/summary"
    rqst = requests.get(api_url)
    print( rqst ) 
    if rqst: 
        rqst = json.loads( rqst.text )

        GLOBAL_DATA = rqst['Global']

        rqst = rqst['Countries']
        for item in rqst:
            if item['Country'] == country:
                KE_DATA = item
    

    morez = ['TotalTests', 'ActiveCases', 'Serious,Critical']
    for m in morez:
        KE_DATA[m] = ''
        GLOBAL_DATA[m] = ''


    return KE_DATA, GLOBAL_DATA


def getLatestSummaryStats_PA(country='Kenya'):
    KE_DATA = None
    GLOBAL_DATA = None

    try:
        KE_DATA, GLOBAL_DATA = getStatsFromWorldometer(country) 

        print( colored( "\n ================= FETCHED FROM WORLDOMETER =================\n", 'red') ) 
    except:
        KE_DATA, GLOBAL_DATA = getStatsFromPublicAPI(country) 
        print( colored( "\n ================= FETCHED FROM PUBLIC API =================\n", 'red') ) 
        traceback.print_exc()
 

    return KE_DATA, GLOBAL_DATA
    
def getRelatedNews():
    NEWS_DATA , NEWS_TICKER = [], []

    api_key = "633d33d95eac415e8334b783cabe3485"
    
    q = "covid19"
    mkt='en-US'
    safeSearch = 'safeSearch'
    news_source = 'medical-news-today' #'google-news' # reuters associated-press bbc-news cnn google-news business-insider bloomberg buzzfeed  etc>>> 
    category = 'health' #&category={category}

    api_url = f"http://newsapi.org/v2/top-headlines?sources={news_source}&apiKey={api_key}"

    rqst = requests.get(api_url)
    rqst = json.loads( rqst.text )
    print(rqst) 
    
    for item in rqst['articles']:
        NEWS_TICKER.append( item['title'] )
        NEWS_DATA.append( item ) 

    return NEWS_DATA , NEWS_TICKER