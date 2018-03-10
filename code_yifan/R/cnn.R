library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 



# read data ---------------------------------------------------------------

data_ori = read_csv("/Users/Lyf/Desktop/train_cnn10w.csv")
feature = read_csv("/Users/Lyf/Github/Stat_628_2/data/feature3_all.csv")
myfeature <- feature %>%
  select(word)
myfeature2 = myfeature[1:5000,]
data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
n1 = length(data_df$line)

tidy_all <- data_df %>%
  unnest_tokens(word, text)
tidy_all2 <- tidy_all %>%
  inner_join(myfeature2, by = "word")

tidy_all3 <- tidy_all2 %>%
  group_by(line) %>%
  summarise(text = paste(word, collapse=" "))


final_data = data_ori %>%
  select(-text)
final_data$line = c(1:nrow(final_data))

final_data <- final_data %>%
  inner_join(tidy_all3, by = "line")


write_csv(final_data, "/Users/Lyf/Desktop/train_cnn10w2.csv")



tidy_all_sp <- tidy_all2 %>%
  group_by(line) %>%
  summarise(count = n())



# test --------------------------------------------------------------------

data_ori = read_csv("/Users/Lyf/Desktop/cnn_test.csv")
feature = read_csv("/Users/Lyf/Github/Stat_628_2/data/feature3_all.csv")
myfeature <- feature %>%
  select(word)
myfeature2 = myfeature[1:5000,]
data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
n1 = length(data_df$line)

tidy_all <- data_df %>%
  unnest_tokens(word, text)
tidy_all2 <- tidy_all %>%
  inner_join(myfeature2, by = "word")

tidy_all3 <- tidy_all2 %>%
  group_by(line) %>%
  summarise(text = paste(word, collapse=" "))


final_data = data_ori %>%
  select(-text)
final_data$line = c(1:nrow(final_data))

final_data <- final_data %>%
  inner_join(tidy_all3, by = "line")


write_csv(final_data, "/Users/Lyf/Desktop/train_cnn10w2.csv")
