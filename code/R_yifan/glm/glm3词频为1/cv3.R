library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 


# read data ---------------------------------------------------------------

data_ori = read_csv("../../data/random100000w.csv")
data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)


# parameter ---------------------------------------------------------------

my_sh = c(1.5, 2.5, 3.5 ,4.5)
var_sh = 0.001
n_sh = 100

# CV ----------------------------------------------------------------------

cv_n = 5
cv_df = tibble(index = c(1:cv_n))
set.seed(615)
cv_index = sample_n(cv_df, nrow(data_df), replace = T)

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

# main --------------------------------------------------------------------



pred_ori = numeric(nrow(cv_index))
my_sh_all = matrix(0, cv_n, 4)
for(ii in 1:cv_n){
  split <- which(cv_index$index != ii)
  train <- data_df[split,]
  test <- data_df[-split,]
  tidy_train <- train %>%
    unnest_tokens(word, text)
  tidy_test <- test %>%
    unnest_tokens(word, text)
  
  tidy_train2 <- tidy_train %>%
    group_by(line, stars, word) %>%
    summarise(mycount = 1)
  
  # delete rare word --------------------------------------------------------

  major <- tidy_train2 %>%
    group_by(word) %>%
    summarise(count = n()) %>%
    arrange(desc(count)) %>%
    filter(count > n_sh)
  
  tidy_train3 <- tidy_train2 %>%
    inner_join(major, by = "word")
  
  
  # delete useless word -----------------------------------------------------
  
 
  tidy_train_count <- tidy_train3 %>%
    count(stars)
  tidy_var = tidy_train3 %>%
    group_by(word, stars) %>%
    summarise(count = n()) %>%
    spread(stars, count, fill = 0)
  myvar_test = apply(tidy_var, 1, myvar, counts = tidy_train_count$n)
  word_var <- data_frame(word = tidy_var$word, var = myvar_test) %>%
    arrange(desc(var)) 
  
  k = sum(word_var$var>var_sh)
  
  
  
  
  tidy_all <- data_df %>%
    unnest_tokens(word, text)
  
  tidy_all2 <- tidy_all %>%
    filter(tidy_all$word %in% word_var$word[1:k]) 
  tidy_all3 <- tidy_all2 %>%
    group_by(line, stars, word) %>%
    summarise(mycount = 1)
  
  xx <- tidy_all3 %>%
    group_by(line) %>%
    summarise(kkk = 1)
  
  w = which(!c(1:nrow(data_df))   %in% xx$line)
  
  
  
  star_mode = Mode(train$stars)
  
  my_tidy = tibble(
    line = w,
    stars = rep(star_mode, length(w)),
    word = rep("myWord", length(w)),
    mycount = rep(1, length(w))
  )
  
  tidy_all4 <- bind_rows(tidy_all3, my_tidy) %>%
    arrange(line)
  
  
  # dataframe to matrix -----------------------------------------------------
  
  
  n2 = length(unique(tidy_all4$word)) 
  
  
  all_matrix <- tidy_all4 %>%
    cast_sparse(line, word, mycount)
  
  
  all_matrix2 = cbind(all_matrix, data_df$stars)
  
  
  # fit model ---------------------------------------------------------------
  
  split2 = split
  train2 <- all_matrix2[split2,]
  test2 <- all_matrix2[-split2,]
  true_value = data_df$stars[-split2]
  
  
  fit = glmnet(train2[,1:n2], train2[,n2+1])
  
  cv <- cv.glmnet(train2[,1:n2], train2[,n2+1],nfolds=5)
  pred <- predict(fit, test2[,1:n2],type="response", s=cv$lambda.min)
  pred_ori[-split] = pred
  
  # calculate sh ------------------------------------------------------------
  
  mymse = function(sh){
    pred_new = rep(0, length(pred))
    pred_new[which(pred<sh[1])] = 1
    pred_new[which(pred>=sh[1] & pred<sh[2])] = 2
    pred_new[which(pred>=sh[2] & pred<sh[3])] = 3
    pred_new[which(pred>=sh[3] & pred<sh[4])] = 4
    pred_new[which(pred>=sh[4])] = 5
    MSE = sum((pred_new-true_value)^2)/length(pred_new)
    return(MSE)
  }
  
  my_sh2 = optim(c(1.5,2.5,3.5,4.5),mymse)
  my_sh3 = my_sh2$par
  my_sh_all[ii,] = my_sh3
  print(ii)
}


# mse ---------------------------------------------------------------------

my_sh_final = apply(my_sh_all, 2, mean)
pred_new = mypred(my_sh_final, pred_ori)
mse = sum((pred_new - data_df$stars)^2)/length(data_df$stars)
mse4var[iii] = mse


# feature -----------------------------------------------------------------

data_ori = read_csv("/Users/Lyf/Desktop/train_en3.csv")
data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
data_df1 = data_df[1:500000,]
data_df2 = data_df[(1:500000)+500000,]
data_df3 = data_df[1000001:nrow(data_df),]

t1 = Sys.time()
tidy_all_1 <- data_df1 %>%
  unnest_tokens(word, text)
tidy_all2_1 <- tidy_all_1 %>%
  distinct(line, stars, word) 
print(Sys.time() - t1)

t1 = Sys.time()
tidy_all_2 <- data_df2 %>%
  unnest_tokens(word, text)
tidy_all2_2 <- tidy_all_2 %>%
  distinct(line, stars, word) 
print(Sys.time() - t1)

t1 = Sys.time()
tidy_all_3 <- data_df3 %>%
  unnest_tokens(word, text)
tidy_all2_3 <- tidy_all_3 %>%
  distinct(line, stars, word) 
print(Sys.time() - t1)

tidy_all2 = bind_rows(tidy_all2_1, tidy_all2_2, tidy_all2_3)

# delete rare word --------------------------------------------------------

n_sh = 100

major <- tidy_all2 %>%
  group_by(word) %>%
  summarise(count = n()) %>%
  arrange(desc(count)) %>%
  filter(count > n_sh)


t1 = Sys.time()
tidy_all3 <- tidy_all2 %>%
  inner_join(major, by = "word")
print(Sys.time() - t1)
# write_csv(tidy_all3, "/Users/Lyf/Desktop/train_all3.csv")

# delete useless word -----------------------------------------------------


tidy_all_count <- tidy_all3 %>%
  group_by(stars) %>%
  summarise(n = n())

t1 = Sys.time()
tidy_var = tidy_all3 %>%
  group_by(word, stars) %>%
  summarise(count = n()) %>%
  spread(stars, count, fill = 0)
print(Sys.time() - t1)

t1 = Sys.time()
myvar_test = apply(tidy_var, 1, myvar, counts = tidy_all_count$n)
print(Sys.time() - t1)

word_var <- data_frame(word = tidy_var$word, var = myvar_test) %>%
  arrange(desc(var)) 

var_sh = 0.001
k = sum(word_var$var>var_sh)

feature3_all = word_var[1:k,]

write_csv(feature3_all, "../../data/feature3_all.csv")

