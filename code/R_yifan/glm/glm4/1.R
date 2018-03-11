library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 



# read data ---------------------------------------------------------------

data_ori = read_csv("/Users/Lyf/Github/Stat_628_2/data/random100000w.csv")

data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
n1 = length(data_df$line)

# parameter ---------------------------------------------------------------

my_sh = c(1.5, 2.5, 3.5 ,4.5)
var_sh = 0.001
n_sh = 100

# variance ----------------------------------------------------------------

myvar = function(data, counts){
  data2 = as.numeric(data[2:6])
  for(i in 1:5){
    data2[i] = data2[i]/counts[i]
  }
  data2 = data2/sum(data2)
  return(var(data2))
}

# expect ------------------------------------------------------------------

myexp = function(data, counts){
  data2 = as.numeric(data[2:6])
  for(i in 1:5){
    data2[i] = data2[i]/counts[i]
  }
  data2 = data2/sum(data2)
  return(sum(data2*c(1,2,3,4,5)))
}


# mode fun ----------------------------------------------------------------

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# pred --------------------------------------------------------------------

mypred = function(sh, pred_ori){
  pred_new = rep(0, length(pred_ori))
  pred_new[which(pred_ori<sh[1])] = 1
  pred_new[which(pred_ori>=sh[1] & pred_ori<sh[2])] = 2
  pred_new[which(pred_ori>=sh[2] & pred_ori<sh[3])] = 3
  pred_new[which(pred_ori>=sh[3] & pred_ori<sh[4])] = 4
  pred_new[which(pred_ori>=sh[4])] = 5
  return(pred_new)
}

# add not -----------------------------------------------------------------

add_not = function(x){
  return(paste("not", x[1], sep = ""))
}

# CV ----------------------------------------------------------------------

cv_n = 5
cv_df = tibble(index = c(1:cv_n))
set.seed(615)
cv_index = sample_n(cv_df, nrow(data_df), replace = T)


# variance ----------------------------------------------------------------

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



# mode fun ----------------------------------------------------------------

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# mse ---------------------------------------------------------------------

mymse2 = function(sh, pred_ori){
  pred_new = rep(0, length(pred_ori))
  pred_new[which(pred_ori<sh[1])] = 1
  pred_new[which(pred_ori>=sh[1] & pred_ori<sh[2])] = 2
  pred_new[which(pred_ori>=sh[2] & pred_ori<sh[3])] = 3
  pred_new[which(pred_ori>=sh[3] & pred_ori<sh[4])] = 4
  pred_new[which(pred_ori>=sh[4])] = 5
  return(pred_new)
}

# add not -----------------------------------------------------------------

add_not = function(x){
  return(paste("not", x[1], sep = ""))
}

# main --------------------------------------------------------------------

n_sh = 100
var_sh = 0.01


for(ii in 1:cv_n){
  
  # split data --------------------------------------------------------------
  
  split <- which(cv_index$index != ii)
  train <- data_df[split,]
  test <- data_df[-split,]
  tidy_train <- train %>%
    unnest_tokens(word, text) %>%
    distinct(line, stars, word)
  tidy_test <- test %>%
    unnest_tokens(word, text) %>%
    distinct(line, stars, word)
  
  
  major <- tidy_train %>%
    group_by(word) %>%
    summarise(count = n()) %>%
    arrange(desc(count)) %>%
    filter(count > n_sh)
  
  tidy_train3 <- tidy_train %>%
    inner_join(major, by = "word")
  
  tidy_train_count <- tidy_train3 %>%
    group_by(stars) %>%
    summarise(n = n())
  
  tidy_var = tidy_train3 %>%
    group_by(word, stars) %>%
    summarise(count = n()) %>%
    spread(stars, count, fill = 0)
  
  myvar_test = apply(tidy_var, 1, myvar, counts = tidy_train_count$n)
  myexp_test = apply(tidy_var, 1, myexp, counts = tidy_train_count$n)
  word_var <- data_frame(word = tidy_var$word, 
                         var = myvar_test,
                         exp = myexp_test) %>%
    arrange(desc(var)) 
  
  # var_sh = 0.01
  # k = sum(word_var$var>var_sh)
  major_word <- word_var %>%
    filter(var>var_sh) %>%
    arrange(desc(exp))
  
  
  # mse ---------------------------------------------------------------------
  
  tidy_test2 <- tidy_test %>%
    inner_join(major_word, by = "word")
  
  pred1 <- tidy_test2 %>%
    group_by(line) %>%
    summarise(pred = mean(exp), count = n())
  
  w = which(!test$line %in% pred1$line)
  star_mode = Mode(train$stars)
  pred2 <- data_frame(line = w, pred = star_mode, count = 0)
  
  pred <- bind_rows(pred1, pred2) %>%
    arrange(line)
  
  mymse = function(sh, pred_ori, true_value){
    pred_new = rep(0, length(pred_ori))
    pred_new[which(pred_ori<sh[1])] = 1
    pred_new[which(pred_ori>=sh[1] & pred_ori<sh[2])] = 2
    pred_new[which(pred_ori>=sh[2] & pred_ori<sh[3])] = 3
    pred_new[which(pred_ori>=sh[3] & pred_ori<sh[4])] = 4
    pred_new[which(pred_ori>=sh[4])] = 5
    return(sum((pred_new- true_value)^2))
  }
  
  my_sh = optim(c(1.5,2.5,3.5,4.5),mymse, pred_ori = pred$pred, true_value = test$stars)
  
}




# feature -----------------------------------------------------------------

data_ori = read_csv("/Users/Lyf/Github/Stat_628_2/data/random100000w.csv")
data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
tidy_all <- data_df %>%
  unnest_tokens(word, text)
tidy_all2 <- tidy_all %>%
  distinct(line, stars, word)


major <- tidy_all2 %>%
  group_by(word) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  filter(count > n_sh)

tidy_all3 <- tidy_all2 %>%
  inner_join(major, by = "word")

tidy_all_count <- tidy_all3 %>%
  group_by(stars) %>%
  summarise(n = n())

tidy_var = tidy_all3 %>%
  group_by(word, stars) %>%
  summarise(count = n()) %>%
  spread(stars, count, fill = 0)

myvar_test = apply(tidy_var, 1, myvar, counts = tidy_all_count$n)
myexp_test = apply(tidy_var, 1, myexp, counts = tidy_all_count$n)
word_var <- data_frame(word = tidy_var$word, 
                       var = myvar_test,
                       exp = myexp_test) %>%
  arrange(desc(var)) 

var_sh = 0.01
k = sum(word_var$var>var_sh)






