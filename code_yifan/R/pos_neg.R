library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 

data_ori = read_csv("../../data/li_first100000forR.csv")

data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)

data_df5 <- data_df %>%
  filter(stars == 5)

data_df1 <- data_df %>%
  filter(stars == 1)


myexpect = function(words, data, counts){
  data_useful <- data %>%
    filter(word == words) %>%
    count(stars)
  names(data_useful)[2] = "n"
  my_star = tibble(stars = c(1:5), n = rep(0, 5))
  data_useful2 <- rbind(data_useful, my_star) %>%
    group_by(stars) %>%
    summarise(n = max(n)) %>%
    mutate(xx = n/counts) %>%
    mutate(xx = xx/sum(xx))
  if(is.na(sum(data_useful2$xx*c(1,2,3,4,5)))){
    return(0)
  }else{
    return(sum(data_useful2$xx*c(1,2,3,4,5)))
  }
}

myvar = function(words, data, counts){
  data_useful <- data %>%
    filter(word == words) %>%
    count(stars)
  names(data_useful)[2] = "n"
  my_star = tibble(stars = c(1:5), n = rep(0, 5))
  data_useful2 <- rbind(data_useful, my_star) %>%
    group_by(stars) %>%
    summarise(n = max(n)) %>%
    mutate(xx = n/counts) %>%
    mutate(xx = xx/sum(xx))
  if(is.na(var(data_useful2$xx))){
    return(0)
  }else{
    return(var(data_useful2$xx))
  }
}

n_sh = 0.01*nrow(data_df)
data_tidy <- data_df %>%
  unnest_tokens(word, text)

major <- data_tidy %>%
  count(word, sort = T) %>%
  filter(n > n_sh)

data_tidy2 <- data_tidy %>%
  inner_join(major, by = "word")

word_var <- data_frame(word = major$word, var = rep(0, nrow(major)), expect = rep(0, nrow(major)))
data_tidy_count <- data_tidy2 %>%
  count(stars)
t1 = Sys.time()
word_var$var <- apply(word_var[,1], 1, myvar, 
                      data = data_tidy2, counts = data_tidy_count$nn)
t2 = Sys.time()
print(t2-t1)
word_var$expect <- apply(word_var[,1], 1, myexpect, 
                      data = data_tidy2, counts = data_tidy_count$nn)
t3 = Sys.time()
print(t3-t2)
word_var <- word_var %>%
  arrange(desc(var))
filename = paste("../../data/pos_neg.csv", sep = "")
write_csv(word_var, filename)

n = 1
data_tidy2 <- data_tidy %>%
  filter(stars == 1)  %>%
  group_by(word) %>%
  summarise(nnn = n()) %>%
  arrange(desc(nnn))

filename = paste("../../data/dist", as.character(n), "_ori.csv", sep = "")
write_csv(data_tidy2, filename)
