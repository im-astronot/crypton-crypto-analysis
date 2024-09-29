import tweepy
import requests
from textblob import TextBlob
from decimal import Decimal, ROUND_HALF_UP       
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from markupsafe import Markup
import random
from flask import Flask, render_template, request, make_response, session, jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField
from wtforms.validators import DataRequired
import yfinance as yf
import numpy as np
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta


auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')




def get_top_crypto_data(limit=100):
    params = {
        "limit": limit,
        "tsym": "USD",
        "api_key": CRYPTOCOMPARE_API_KEY
    }
    response = requests.get(CRYPTOCOMPARE_API_URL, params=params)
    data = response.json()["Data"]
    return data

@app.route('/crypto_data', methods=['GET'])
def crypto_data():
    data = get_top_crypto_data()
    return jsonify(data)

@app.route('/historical_data')
def historical_data():
    symbol = request.args.get('symbol')
    ticker = yf.Ticker(f'{symbol}-USD')

    # Get the last 24 hours of data
    now = datetime.now()
    start = now - timedelta(days=1)
    historical_data = ticker.history(start=start, end=now, interval="1m")
    
    # Prepare data for the response
    response_data = {
        'timestamps': historical_data.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'prices': historical_data['Close'].tolist()
    }
    
    return jsonify(response_data)


@app.route('/price')
def price():
    return render_template('price.html')

@app.template_filter('intcomma')
def intcomma_filter(value):
    """
    Adds commas to an integer or float value.
    """
    if isinstance(value, (int, float)):
        parts = []
        value = str(value)
        if '.' in value:
            int_part, dec_part = value.split('.')
            parts.append(int_part)
            parts.append('.')
            parts.append(dec_part)
        else:
            parts.append(value)
        parts[0] = '{:,d}'.format(int(parts[0])).replace(',', '.')
        return Markup(''.join(parts))
    else:
        return value



@app.route('/news')
def news():
    response = requests.get('https://min-api.cryptocompare.com/data/v2/news/', params={
        'lang': 'EN',
        
    })
    data = response.json()

    # Extract the titles, descriptions, and thumbnails of the first 20 articles
    titles = []
    descriptions = []
    thumbnails = []
    links = []
    for article in data['Data'][:20]:  # Get the first 20 articles
        title = article['title']
        description = article['body'][:600] + '...' if len(article['body']) > 600 else article['body']
        thumbnail = article['imageurl']
        link = article['url']
        titles.append(title)
        descriptions.append(description)
        thumbnails.append(thumbnail)
        links.append(link)

    return render_template('news.html', titles=titles, descriptions=descriptions, thumbnails=thumbnails, links=links)


def get_sentiment(crypto):
    # Search for tweets about the given cryptocurrency
    tweets = api.search_tweets(q='#cryptocurrency', lang='en')
    
    # Initialize counters for positive, negative, and neutral tweets
    positive = 0
    negative = 0
    neutral = 0
    
    # Iterate through the tweets and perform sentiment analysis
    for tweet in tweets:
        analysis = TextBlob(tweet.text)
        if analysis.sentiment.polarity > 0:
            positive += 1
        elif analysis.sentiment.polarity < 0:
            negative += 1
        else:
            neutral += 1
            
    # Calculate the total number of tweets
    total = positive + negative + neutral
    
    # Calculate the percentage of positive, negative, and neutral tweets
    positive_percent = positive / total * 100
    negative_percent = negative / total * 100
    neutral_percent = neutral / total * 100
    
    # Return the results as a dictionary
    return {
        'positive': positive_percent,
        'negative': negative_percent,
        'neutral': neutral_percent
    }

def show_sentiment_chart(sentiment):
    # Extract the positive, negative, and neutral percentages from the sentiment dictionary
    positive = sentiment['positive']
    negative = sentiment['negative']
    neutral = sentiment['neutral']

    # Set up the bar chart
    data = [go.Bar(
                x=['Positive', 'Negative', 'Neutral'],
                y=[positive, negative, neutral],
                marker=dict(color=['green', 'red', 'blue'])
           )]
    layout = go.Layout(title='Sentiment Analysis', yaxis=dict(title='Percentage'))
    fig = go.Figure(data=data, layout=layout)

    # Return the chart object
    return fig

@app.route('/sentiment')
def sentiment():
    # Get the cryptocurrency from the request query string
    crypto = request.args.get('crypto')
    
    # Perform sentiment analysis on the cryptocurrency
    sentiment = get_sentiment(crypto)
    
    # Generate the sentiment chart
    show_sentiment_chart(sentiment)
    
    # Render the template and pass the sentiment data to it
    return render_template('sentiment.html', crypto=crypto, sentiment=sentiment)


@app.route('/tweets')
def get_tweets():
    # Search for tweets about cryptocurrency
    tweets = api.search_tweets(q='#cryptocurrency', lang='en')
    
    # Extract the text and time of the first 10 tweets
    tweet_data = []
    
    for tweet in tweets[:10]:  # Get the first 10 tweets
        tweet_text = tweet.text
        tweet_time = tweet.created_at.strftime('%d/%m/%Y %H:%M:%S')
        tweet_data.append({'text': tweet_text, 'time': tweet_time})

    # Render the tweets.html template and pass in the tweet data
    return render_template('tweets.html', tweets=tweet_data)


@app.route('/convert')
def convert():
    return render_template('convert.html')

@app.route('/convert', methods=['POST'])
def result():
    # Get the form data
    amount = Decimal(request.form['amount'])
    currency = request.form['currency']

    # Make a request to the CoinMarketCap API
    r = requests.get(f'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest', headers={'X-CMC_PRO_API_KEY': '5fde57d8-5794-480e-9d2d-32ea42aacfbc'}, params={'symbol': 'BTC', 'convert': currency})
    data = r.json()

    # Calculate the conversion result
    rate = Decimal(data['data']['BTC']['quote'][currency]['price'])
    result = amount * rate
    
    # Round off the result to 2 decimal places
    result = result.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    # Pass the input values and result to the template
    return render_template('convert.html', result=result, amount=amount, currency=currency)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        selected_coin = 'BTC'
    elif request.method == 'POST':
        selected_coin = request.form.get('coin').upper()

    days = 10
    url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={selected_coin}&tsym=USD&limit={days}'

    response = requests.get(url)
    data = json.loads(response.text)
    prices = data['Data']['Data']
    df = pd.DataFrame(prices, columns=['time', 'low', 'high', 'open', 'close', 'volume', 'conversionType', 'conversionSymbol'])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[['open']]

    # Compute the moving average
    rolling_mean = df.rolling(window=2).mean()

    # Train the Random Forest model
    X_train = df.index.values.astype(float)[:, None]
    y_train = df['open']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make a prediction for the next day
    last_date = df.index[-1]
    next_date = last_date + pd.DateOffset(days=1)
    next_date_str = next_date.strftime('%d/%m/%Y')
    next_date_float = next_date.value / 10**9
    prediction = model.predict([[next_date_float]])[0]
    predicted_price_str = f'${prediction:,.2f}'

    # Evaluate model performance using MAE, MSE, and R2
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    print('MAE:', mae)
    print('MSE:', mse)
    print('R2:', r2)

    # Generate a line graph using Plotly Express
    title = f'{selected_coin} Price Prediction'
    fig = px.line(df, x=df.index, y='open', labels={'x': 'Date', 'y': 'Price in USD'}, title=title)
    fig.add_scatter(x=df.index, y=rolling_mean['open'], mode='lines', line=dict(color='orange', width=2), name='Moving Average')
    fig.add_scatter(x=[next_date], y=[prediction], mode='markers', marker=dict(color='green', size=10), name='Predicted Price')
    fig.update_yaxes(tickformat="$,.2f")
    fig.update_layout(
        xaxis_title='Price',
        yaxis_title='Date'
    )
    fig.update_traces(line=dict(color='blue', width=2), selector=dict(type='scatter', mode='lines', name='Price'))
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('prediction.html', plot=graphJSON, predicted_price=predicted_price_str, predicted_date=next_date_str, selected_coin=selected_coin)


#Portfolio
app.config['SECRET_KEY'] = 'your_secret_key'

class CryptoForm(FlaskForm):
    asset_id = StringField('Asset', validators=[DataRequired()])
    quantity = DecimalField('Quantity', validators=[DataRequired()])
    purchase_price = DecimalField('Purchase Price', validators=[DataRequired()])

@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    form = CryptoForm()

    if not session.get('portfolio_data'):
        session['portfolio_data'] = []

    portfolio_data = session['portfolio_data']
    
    if request.method == "POST" and form.validate():
        asset_id = form.asset_id.data
        quantity = form.quantity.data
        purchase_price = form.purchase_price.data

        lower_bound = purchase_price * Decimal('0.8')
        upper_bound = purchase_price * Decimal('1.2')
        current_price = random.uniform(float(lower_bound), float(upper_bound))
        value = current_price * float(quantity)

        new_item = {
            "asset": asset_id,
            "quantity": str(quantity),
            "purchase_price": str(purchase_price),
            "current_price": current_price,
            "value": value
        }
        portfolio_data.append(new_item)

        # Update the session data
        session['portfolio_data'] = portfolio_data

    total_value = round(sum(item["value"] for item in portfolio_data), 2)

    return render_template('portfolio.html', form=form, portfolio_data=portfolio_data, total_value=total_value)

@app.route('/download', methods=['GET'])
def download():
    portfolio_data = session.get('portfolio_data', [])

    json_data = json.dumps(portfolio_data, indent=4)

    response = make_response(json_data)
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = 'attachment; filename=portfolio.json'

    return response

# Trade Signal
@app.route('/tradesignal')
def tradesig():
    return render_template('signal.html')

@app.route('/get_signal', methods=['POST'])
def get_signal():
    symbol = 'BTC-USD'
    signal = generate_trade_signal(symbol)
    return {'signal': signal}

@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        symbol = 'BTC-USD'
        data = fetch_data(symbol)
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
        json_data = data.to_json(orient='records', date_format='iso', date_unit='s')
        json_data = json_data.replace('NaN', 'null')
        return jsonify(json.loads(json_data))
    except Exception as e:
        app.logger.exception(e)
        return {"error": str(e)}, 500


def generate_trade_signal(symbol, short_window=5, long_window=20):
    data = fetch_data(symbol)
    signals = moving_average_crossover(data, short_window, long_window)

    if signals.iloc[-1]['Signal'] == 1:
        return 'Buy'
    elif signals.iloc[-1]['Signal'] == -1:
        return 'Sell'
    else:
        return 'Hold'

def fetch_data(symbol, period='1y'):
    data = yf.download(symbol, period=period)
    add_technical_indicators(data)
    return data

def add_technical_indicators(data):
    # Add Bollinger Bands
    bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
    data["BB_High"] = bb.bollinger_hband()
    data["BB_Low"] = bb.bollinger_lband()

    # Add RSI
    rsi = RSIIndicator(close=data["Close"], window=14)
    data["RSI"] = rsi.rsi()

def moving_average_crossover(data, short_window, long_window): #with technical indicators
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    signals['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    signals['Signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1.0, -1.0)
    signals['Positions'] = signals['Signal'].diff()

    return signals












































@app.route('/about')
def about():
    return render_template('aboutus.html')


if __name__ == '__main__':
    app.run(debug=True)
