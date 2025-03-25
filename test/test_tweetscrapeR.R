library(TweetScraperR)
userPass <- readLines('E:/Users/1167486/Local/test_pass.txt')
user<-as.character(userPass[1])
pass <-as.character(userPass[2])

setwd('E:/Users/1167486/Local/data')

openTwitter(view=TRUE)


tweetUrl <- 'https://x.com/f_solorzano/status/1896405011846098960'
tweetUrl2 <- 'https://x.com/RicardoBSalinas/status/1900807572610818356'

getTweetsReplies(url = tweetUrl, 
                 xuser = user, 
                 xpass= pass,
                 n_tweets = 100,
                 dir = getwd())




#getTweetsReplies(url = tweetUrl2, xuser = user, xpass= pass)

#x1 <- getTweetsReplies('https://x.com/RicardoBSalinas/status/1900807572610818356')


#getTweetsHistoricalSearch(search = "narco")

# getTweetsHistoricalHashtag(
#   hashtag = "#8M",
#   timeout = 10,
#   n_tweets = 100,
#   since = "2025-03-07",
#   until = "2025-03-09",
#   xuser = user,
#   xpass = pass,
#   dir = getwd()
# )
