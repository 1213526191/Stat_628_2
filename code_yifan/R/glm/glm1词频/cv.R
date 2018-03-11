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

# my_sh = c(1.45, 2.92, 3.6, 4.15)
my_sh = c(1.5, 2.5, 3.5 ,4.5)

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


var_sh_all = 0.001
NN = length(var_sh_all)
mse4var = numeric(NN)
for(iii in 1:NN){
  
  # split data --------------------------------------------------------------
  
  print(iii)
  var_sh = var_sh_all[iii]
  
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
    
    # not dict ----------------------------------------------------------------
    
    ## check stop words important or not
    
    var_not = data_frame(word = stop_words$word, var = 0)
    count <- tidy_train %>%
      count(stars)
    
    var_not$var = apply(stop_words[,1], 1, myvar, data = tidy_train, counts = count$n)
    
    index_not = which(var_not$var>var_sh)
    my_stop = stop_words[-index_not,]
    my_stop2 <- my_stop
    my_stop2$word = apply(my_stop, 1, add_not)
    
    # delete stop word --------------------------------------------------------
    
    
    
    tidy_train2 <- tidy_train %>%
      anti_join(my_stop, by = "word") %>%
      anti_join(my_stop2, by = "word")
    
    
    # delete rare word --------------------------------------------------------
    
    n_sh = 0.01 * nrow(train)
    major <- tidy_train2 %>%
      count(word, sort = T) %>%
      filter(n > n_sh)
    
    tidy_train3 <- tidy_train2 %>%
      inner_join(major, by = "word")
    
    
    # delete useless word -----------------------------------------------------
    
    word_var <- data_frame(word = major$word, var = rep(0, nrow(major)))
    tidy_train_count <- tidy_train3 %>%
      count(stars)
    word_var$var <- apply(word_var[,1], 1, myvar, 
                          data = tidy_train3, counts = tidy_train_count$nn)
    
    word_var <- word_var %>%
      arrange(desc(var))
    
    k = sum(word_var$var>var_sh)
    
    
    
    
    tidy_all <- data_df %>%
      unnest_tokens(word, text)
    
    tidy_all2 <- tidy_all %>%
      filter(tidy_all$word %in% word_var$word[1:k]) %>%
      mutate(count = 1)
    
    xx <- tidy_all2 %>%
      group_by(line) %>%
      summarise(star = mean(stars))
    
    w = which(!c(1:nrow(data_df))   %in% xx$line)
    
    
    
    star_mode = Mode(train$stars)
    
    my_tidy = tibble(
      line = w,
      stars = rep(star_mode, length(w)),
      word = rep("myWord", length(w)),
      count = rep(1, length(w))
    )
    
    tidy_all3 <- full_join(tidy_all2, my_tidy, by = c("line", "stars", "word", "count")) %>%
      arrange(line)
    
    
    # dataframe to matrix -----------------------------------------------------
    
    
    n2 = length(unique(tidy_all3$word)) 
    
    all_sp <- tidy_all3 %>%
      cast_dtm(line, word, count)
    
    all_matrix <- tidy_all3 %>%
      cast_sparse(line, word, count)
    
    # inspect(all_sp[20, 1:20])
    
    # all_matrix = dtm.to.Matrix(all_sp)
    
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
  
  
  pred_new = mypred(my_sh, pred_ori)
  mse = sum((pred_new - data_df$stars)^2)/length(data_df$stars)
  mse4var[iii] = mse
}
sh_final = apply(my_sh_all, 2, mean)
pred_new2 = mypred(sh_final, pred_ori)
mse = sum((pred_new2 - data_df$stars)^2)/length(data_df$stars)

# # result table ------------------------------------------------------------
# 
# result = tibble(
#   pred = pred_new,
#   true = data_df$stars
# )
# result_ma = tibble(
#   a = rep(0, 5),
#   b = rep(0, 5),
#   c = rep(0, 5),
#   d = rep(0, 5),
#   e = rep(0, 5)
# )
# for(i in 1:5){
#   aaa <- result %>%
#     filter(true == i) %>%
#     group_by(pred) %>%
#     summarise(n = n())
#   bbb = tibble(pred = c(1:5), n = rep(0, 5))
#   ccc <- rbind(aaa, bbb) %>%
#     group_by(pred) %>%
#     summarise(n = max(n)) 
#   result_ma[i,] = ccc$n
# }
# # system("say 完成")
# 
# result_ma$accucury = 0
# for(i in 1:5){
#   result_ma$accucury[i] = as.numeric(result_ma[i,i]/sum(result_ma[i, 1:5]))
# }
# result_ma
# 
# 
# mse_sp = rep(0,5)
# for(i in 1:5){
#   cos = c(1:5)
#   mse_sp[i] = sum((cos-i)^2*result_ma[i,1:5])
# }
# mse_sp


# feature -----------------------------------------------------------------

tidy_all <- data_df %>%
  unnest_tokens(word, text)


var_not = data_frame(word = stop_words$word, var = 0)
count <- tidy_all %>%
  count(stars)

var_not$var = apply(stop_words[,1], 1, myvar, data = tidy_all, counts = count$n)

index_not = which(var_not$var>var_sh)
my_stop = stop_words[-index_not,]
my_stop2 <- my_stop
my_stop2$word = apply(my_stop, 1, add_not)

tidy_all2 <- tidy_all %>%
  anti_join(my_stop, by = "word") %>%
  anti_join(my_stop2, by = "word")


# delete rare word --------------------------------------------------------

n_sh = 0.01 * nrow(data_df)
major <- tidy_all2 %>%
  count(word, sort = T) %>%
  filter(n > n_sh)


tidy_all3 <- tidy_all2 %>%
  inner_join(major, by = "word")


# delete useless word -----------------------------------------------------

word_var <- data_frame(word = major$word, var = rep(0, nrow(major)))
tidy_train_count <- tidy_all3 %>%
  count(stars)
word_var$var <- apply(word_var[,1], 1, myvar, 
                      data = tidy_all3, counts = tidy_train_count$nn)

word_var <- word_var %>%
  arrange(desc(var))

k = sum(word_var$var>var_sh)

feature = word_var[1:k,]

write_csv(feature, "../../data/feature.csv")








