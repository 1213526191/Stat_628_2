library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 

data_ori = read_csv("/Users/Lyf/OneDrive/study/WISC/2017_spring/Stat_628/hw2/trash/train_en3.csv")

data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
data_df1 = data_df[1:500000,]
data_df2 = data_df[500001:1000000,]
data_df3 = data_df[1000001:nrow(data_ori),]
t1 = Sys.time()
data_tidy1 <- data_df1 %>%
  unnest_tokens(word, text)
t2 = Sys.time()
t2-t1
t1 = Sys.time()
data_tidy2 <- data_df2 %>%
  unnest_tokens(word, text)
t2 = Sys.time()
t2-t1
t1 = Sys.time()
data_tidy3 <- data_df3 %>%
  unnest_tokens(word, text)
t2 = Sys.time()
t2-t1

#####

t1 = Sys.time()
data_tidy11 <- data_tidy1 %>%
  distinct(line, word, stars)
t2 = Sys.time()
t2-t1
t1 = Sys.time()
data_tidy21 <- data_tidy2 %>%
  distinct(line, word, stars)
t2 = Sys.time()
t2-t1
t1 = Sys.time()
data_tidy31 <- data_tidy3 %>%
  distinct(line, word, stars)
t2 = Sys.time()
t2-t1

#######

t2-t1
data_tidy <- bind_rows(data_tidy11, data_tidy21, data_tidy31)

data_tidy2 <- data_tidy %>%
  group_by(stars, word) %>%
  summarise(count = n()) 
data_tidy2_5 <- data_tidy2 %>%
  group_by(word) %>%
  summarise(count2_5 = sum(count)) %>%
  filter(count2_5 > 100) 
major <- data_tidy2_5 %>%
  select(word)
data_tidy3 <- data_tidy2 %>%
  inner_join(major, by = "word") %>%
  spread(stars, count, fill = 0)

star_count <- data_df %>%
  group_by(stars) %>%
  summarise(count = n())
  
mycount = star_count$count
t2 = Sys.time()
t2-t1

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
t1 = Sys.time()
result = as.tibble(t(apply(data_tidy3, 1, myprop, counts = mycount)))
t2 = Sys.time()
t2-t1
result$word = data_tidy3$word
result <- result %>%
  select(word, V1:V5)


filename = paste("feature4jianmin_all.csv")
write_csv(result, filename)

result %>%
  arrange(desc(V1))

data_tidy3[which(data_tidy3$word %in% "refund"),][2:6]/mycount
result[which(result$word %in% "and"),]



