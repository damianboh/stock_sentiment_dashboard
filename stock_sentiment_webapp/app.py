import flask
from flask import Flask, render_template # for web app
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import plotly
import plotly.express as px
import json # for graph plotting in website
# NLTK VADER for sentiment analysis
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# for extracting data from finviz
finviz_url = 'https://finviz.com/quote.ashx?t='

def get_news(ticker):
    url = finviz_url + ticker
    req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}) 
    response = urlopen(req)    
    # Read the contents of the file into 'html'
    html = BeautifulSoup(response)
    # Find 'news-table' in the Soup and load it into 'news_table'
    news_table = html.find(id='news-table')
    return news_table
	
# parse news into dataframe
def parse_news(news_table):
    parsed_news = []
    
    for x in news_table.findAll('tr'):
        # read the text from each tr tag into text
        # get text from a only
        text = x.a.get_text() 
        # splite text in the td tag into a list 
        date_scrape = x.td.text.split()
        # if the length of 'date_scrape' is 1, load 'time' as the only element

        if len(date_scrape) == 1:
            time = date_scrape[0]
            
        # else load 'date' as the 1st element and 'time' as the second    
        else:
            date = date_scrape[0]
            time = date_scrape[1]
        
        # Append ticker, date, time and headline as a list to the 'parsed_news' list
        parsed_news.append([date, time, text])
        
        # Set column names
        columns = ['date', 'time', 'headline']

        # Convert the parsed_news list into a DataFrame called 'parsed_and_scored_news'
        parsed_news_df = pd.DataFrame(parsed_news, columns=columns)
        
        # Create a pandas datetime object from the strings in 'date' and 'time' column
        parsed_news_df['datetime'] = pd.to_datetime(parsed_news_df['date'] + ' ' + parsed_news_df['time'])
        
    return parsed_news_df
        
def score_news(parsed_news_df):
    # Instantiate the sentiment intensity analyzer
    vader = SentimentIntensityAnalyzer()
    
    # Iterate through the headlines and get the polarity scores using vader
    scores = parsed_news_df['headline'].apply(vader.polarity_scores).tolist()

    # Convert the 'scores' list of dicts into a DataFrame
    scores_df = pd.DataFrame(scores)

    # Join the DataFrames of the news and the list of dicts
    parsed_and_scored_news = parsed_news_df.join(scores_df, rsuffix='_right')
    
            
    parsed_and_scored_news = parsed_and_scored_news.set_index('datetime')
    
    parsed_and_scored_news = parsed_and_scored_news.drop(['date', 'time'], 1)    
        
    parsed_and_scored_news = parsed_and_scored_news.rename(columns={"compound": "sentiment_score"})

    return parsed_and_scored_news

def plot_hourly_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('H').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Hourly Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

def plot_daily_sentiment(parsed_and_scored_news, ticker):
   
    # Group by date and ticker columns from scored_news and calculate the mean
    mean_scores = parsed_and_scored_news.resample('D').mean()

    # Plot a bar chart with plotly
    fig = px.bar(mean_scores, x=mean_scores.index, y='sentiment_score', title = ticker + ' Daily Sentiment Scores')
    return fig # instead of using fig.show(), we return fig and turn it into a graphjson object for displaying in web page later

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods = ['POST'])
def sentiment():

	ticker = flask.request.form['ticker'].upper()
	news_table = get_news(ticker)
	parsed_news_df = parse_news(news_table)
	parsed_and_scored_news = score_news(parsed_news_df)
	fig_hourly = plot_hourly_sentiment(parsed_and_scored_news, ticker)
	fig_daily = plot_daily_sentiment(parsed_and_scored_news, ticker)

	graphJSON_hourly = json.dumps(fig_hourly, cls=plotly.utils.PlotlyJSONEncoder)
	graphJSON_daily = json.dumps(fig_daily, cls=plotly.utils.PlotlyJSONEncoder)
	
	header= "Hourly and Daily Sentiment of {} Stock".format(ticker)
	description = """
	The above chart averages the sentiment scores of {} stock hourly and daily.
	The table below gives each of the most recent headlines of the stock and the negative, neutral, positive and an aggregated sentiment score.
	The news headlines are obtained from the FinViz website.
	Sentiments are given by the nltk.sentiment.vader Python library.
    """.format(ticker)
	return render_template('sentiment.html',graphJSON_hourly=graphJSON_hourly, graphJSON_daily=graphJSON_daily, header=header,table=parsed_and_scored_news.to_html(classes='data'),description=description)



        
if __name__ == '__main__':
    app.run()