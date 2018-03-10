library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 

data_ori = read_csv("../../data/random100000w.csv")

data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
data_tidy <- data_df %>%
  unnest_tokens(word, text)




data_tidy2 <- data_tidy %>%
  distinct(line, word, stars) %>%
  group_by(stars, word) %>%
  summarise(count = n()) 
data_tidy2_5 <- data_tidy2 %>%
  group_by(word) %>%
  summarise(count2_5 = sum(count)) %>%
  filter(count2_5 > 50) 
major <- data_tidy2_5 %>%
  select(word)
data_tidy3 <- data_tidy2 %>%
  inner_join(major, by = "word") %>%
  spread(stars, count, fill = 0)

star_count <- data_df %>%
  group_by(stars) %>%
  summarise(count = n())
  
mycount = star_count$count

myprop = function(data_row, counts){
  data2 = as.numeric(data_row[2:6])
  xx = numeric(5)
  for(i in 1:5){
    if(counts[i] == 0){
      xx[i] = 0
    }else if(sum(data2[-i]) == 0){
      xx[i] = 100
    }else if(sum(counts[-i]) == 0){
      xx[i] = 100
    }else{
      y1 = data2[i]/counts[i]
      y2 = sum(data2[-i])/sum(counts[-i])
      xx[i] = y1/y2
    }
  }
  return(xx)
}

result = as.tibble(t(apply(data_tidy3, 1, myprop, counts = mycount)))
result$word = data_tidy3$word
result <- result %>%
  select(word, V1:V5)


filename = paste("../../data/feature4jianmin.cav")
write_csv(result, filename)
