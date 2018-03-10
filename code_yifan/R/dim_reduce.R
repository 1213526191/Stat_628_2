library(neuralnet)
library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 


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


# main --------------------------------------------------------------------

data_ori = read_csv("/Users/Lyf/Github/Stat_628_2/data/random100000w.csv")
data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
cv_n = 5
cv_df = tibble(index = c(1:cv_n))
set.seed(615)
cv_index = sample_n(cv_df, nrow(data_df), replace = T)
pred_all = rep(0, nrow(data_df))
my_sh_all = matrix(0, cv_n ,4)
for(ii in 1:cv_n){
  t1 = Sys.time()
  split <- which(cv_index$index != ii)
  train <- data_df[split,]
  test <- data_df[-split,]
  
  
  tidy_train <- train %>%
    unnest_tokens(word, text)
  tidy_train2 <- tidy_train %>%
    distinct(line, stars, word)
  tidy_test <- test %>%
    unnest_tokens(word, text)
  tidy_test2 <- tidy_test %>%
    distinct(line, stars, word)
  
  
  major <- tidy_train2 %>%
    group_by(word) %>%
    summarise(count = n()) %>%
    arrange(desc(count)) %>%
    filter(count > n_sh)
  
  tidy_train3 <- tidy_train2 %>%
    inner_join(major, by = "word")
  tidy_test3 <- tidy_test2 %>%
    inner_join(major, by = "word")
  
  tidy_train_count <- tidy_train3 %>%
    group_by(stars) %>%
    summarise(n = n())
  
  tidy_var = tidy_train3 %>%
    group_by(word, stars) %>%
    summarise(count = n()) %>%
    spread(stars, count, fill = 0)
  
  myvar_test = apply(tidy_var, 1, myvar, counts = tidy_train_count$n)
  # myexp_test = apply(tidy_var, 1, myexp, counts = tidy_all_count$n)
  word_var <- data_frame(word = tidy_var$word, 
                         var = myvar_test) %>%
    arrange(desc(var)) 
  
  
  k = sum(word_var$var>var_sh)
  major_word <- word_var %>%
    filter(var>var_sh) %>%
    select(word)
  
  
  
  tidy_train4 <- tidy_train3 %>%
    select(line, stars, word) %>%
    inner_join(major_word, by = "word") %>%
    mutate(mycount = 1)
  tidy_test4 <- tidy_test3 %>%
    select(line, stars, word) %>%
    inner_join(major_word, by = "word") %>%
    mutate(mycount = 1)
  
  xx <- tidy_train4 %>%
    group_by(line) %>%
    summarise(kkk = 0)
  w = which(!train$line   %in% xx$line)
  ww = train$line[w]
  star_mode = round(mean(train$stars))
  my_tidy = tibble(
    line = ww,
    stars = star_mode,
    word = "myWord",
    mycount = 1
  )
  
  tidy_train5 <- bind_rows(tidy_train4, my_tidy) %>%
    arrange(line) 
  
  xx <- tidy_test4 %>%
    group_by(line) %>%
    summarise(kkk = 0)
  w = which(!test$line   %in% xx$line)
  ww = test$line[w]
  my_tidy = tibble(
    line = ww,
    stars = star_mode,
    word = "myWord",
    mycount = 1
  )
  
  tidy_test5 <- bind_rows(tidy_test4, my_tidy) %>%
    arrange(line) 
  
  col_name = unique(tidy_train5$word)
  col_name2 = c(col_name, "STAR")
  train_matrix <- tidy_train5 %>%
    cast_sparse(line, word, mycount)
  stars = train$stars
  train_matrix2 = cbind(train_matrix, stars)
  train_df = as.data.frame(as.matrix(train_matrix2))
  colnames(train_df) = col_name2
  
  test_matrix <- tidy_test5 %>%
    cast_sparse(line, word, mycount)
  
  test_matrix2 = test_matrix
  test_df = as.data.frame(as.matrix(test_matrix2))
  colnames(test_df) = col_name[1:ncol(test_df)]

  f <- as.formula(paste("STAR ~", paste(sprintf("`%s`", col_name[1:(length(col_name)-1)]), #
                                        collapse = " + ")))
  fit = neuralnet(formula = f, data = train_df, hidden=c(5,3)) # , hidden=c(5,3)
  pred = neuralnet::compute(fit, test_df[1:(length(col_name)-1)])
  pred_all[-split] = pred$net.result
  
  mymse = function(sh, preds, true_value){
    pred_new = rep(0, length(pred))
    pred_new[which(preds<sh[1])] = 1
    pred_new[which(preds>=sh[1] & preds<sh[2])] = 2
    pred_new[which(preds>=sh[2] & preds<sh[3])] = 3
    pred_new[which(preds>=sh[3] & preds<sh[4])] = 4
    pred_new[which(preds>=sh[4])] = 5
    MSE = sum((pred_new-true_value)^2)/length(pred_new)
    return(MSE)
  }
  my_sh2 = optim(c(1.5,2.5,3.5,4.5),mymse, preds = pred$net.result, true_value = test$stars)
  my_sh3 = my_sh2$par
  my_sh_all[ii,] = my_sh3
  
  t2 = Sys.time()
  print(paste(as.character(ii), ": ", (t2 - t1)), sep = "")
}
write_csv(pred_all, "pred_all.csv")
write_csv(my_sh_all, "my_sh_all.csv")

# final -------------------------------------------------------------------

library(RTextTools)
library(lsa)
library(rm)
data_ori = read_csv("/Users/Lyf/Github/Stat_628_2/data/random100000w.csv")

data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
tidy_all <- data_df %>%
  unnest_tokens(word, text)
tdm=create_matrix(tidy_all,removeNumbers=T)
tdm_tfidf=weightTfIdf(tdm)
m=as.matrix(tdm)
m_tfidf=as.matrix(tdm_tfidf)



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
# myexp_test = apply(tidy_var, 1, myexp, counts = tidy_all_count$n)
word_var <- data_frame(word = tidy_var$word, 
                       var = myvar_test) %>%
  arrange(desc(var)) 


k = sum(word_var$var>var_sh)
major_word <- word_var %>%
  filter(var>var_sh) %>%
  select(word)

  

tidy_all4 <- tidy_all3 %>%
  select(line, stars, word) %>%
  inner_join(major_word, by = "word") %>%
  mutate(mycount = 1)

xx <- tidy_all4 %>%
  group_by(line) %>%
  summarise(kkk = 0)
w = which(!data_df$line   %in% xx$line)

star_mode = round(mean(data_df$stars))
my_tidy = tibble(
  line = w,
  stars = star_mode,
  word = "myWord",
  mycount = 1
)

tidy_all5 <- bind_rows(tidy_all4, my_tidy) %>%
  arrange(line) 

col_name = unique(tidy_all5$word)
col_name2 = c(col_name, "STAR")
all_matrix <- tidy_all5 %>%
  cast_sparse(line, word, mycount)
stars = data_df$stars
all_matrix2 = cbind(all_matrix, stars)
colnames(all_matrix2) = col_name2

t1 = Sys.time()
all_df = as.data.frame(as.matrix(all_matrix2))
print(Sys.time() - t1)

t1 = Sys.time()
f <- as.formula(paste("STAR ~", paste(sprintf("`%s`", col_name[1:(length(col_name)-1)]),
                                      collapse = " + ")))
fit = neuralnet(formula = f, data = all_df, hidden=c(5,3)) # , hidden=c(5,3)
save(fit, "fit.RData")
print(Sys.time() - t1)




