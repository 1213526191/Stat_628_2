library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 


#  read data --------------------------------------------------------------

data_ori = read_csv("../../data/random100000.csv")
test_ori = read_csv("/Users/Lyf/OneDrive/study/WISC/2017_spring/Stat_628/hw2/test.csv")

train = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
test = data_frame(line = 1:nrow(test_ori),
                     text = test_ori$text, 
                     stars = 0)
n1 = length(data_df$line)

# feature -----------------------------------------------------------------

features = read_csv("../../data/pos_neg.csv")
NNN = sum(features$var > 0.001)
major <- features[1:NNN,] %>%
  select(word)

# threshold ---------------------------------------------------------------

my_sh = c(1.45, 2.92, 3.6, 4.15)


# predict -----------------------------------------------------------------

t1 = Sys.time()
tidy_train <- train %>%
  unnest_tokens(word, text)
t2 = Sys.time()
print(t2-t1)
tidy_test <- test %>%
  unnest_tokens(word, text)
t3 = Sys.time()
print(t3-t2)

tidy_train <- tidy_train %>%
  inner_join(major, by = "word")



