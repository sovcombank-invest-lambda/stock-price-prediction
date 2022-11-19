import feedparser

# Moscow exchange
RSS_FEEDS = "https://www.moex.com/export/news.aspx?cat=200"

# get last news via rss
def get_news():
    feed = feedparser.parse(RSS_FEEDS)
    if feed['bozo'] == 1:
        print("Error: RSS feed is not available")
        return None
    first_article = feed['entries'][0]
    print(first_article['title'])
    print(first_article['summary'])
    print(first_article['link'])

get_news()