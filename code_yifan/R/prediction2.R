# 这个是tf idf 之后的

library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 



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

#  read data --------------------------------------------------------------

train_ori = read_csv("../../data/random100000w.csv")
test_ori = read_csv("/Users/Lyf/OneDrive/study/WISC/2017_spring/Stat_628/hw2/test.csv")

train = data_frame(line = 1:nrow(train_ori),
                     text = train_ori$text, 
                     stars = train_ori$stars)
test = data_frame(line = test_ori$Id,
                     text = test_ori$text, 
                     stars = 0)
n1 = nrow(test)

# feature -----------------------------------------------------------------

features = read_csv("../../data/feature.csv")
major <- features %>%
  select(word)

# threshold ---------------------------------------------------------------

# my_sh = c(1.45, 2.92, 3.6, 4.15)
my_sh = c(1.314392, 3.028014, 3.541586, 4.148861)


# fit model ---------------------------------------------------------------


t1 = Sys.time()
tidy_train <- train %>%
  unnest_tokens(word, text)
t2 = Sys.time()
print(t2-t1)
t2 = Sys.time()
tidy_test <- test %>%
  unnest_tokens(word, text)
t3 = Sys.time()
print(t3-t2)

tidy_train2 <- tidy_train %>%
  inner_join(major, by = "word") %>%
  mutate(count = 1)
xx <- tidy_train2 %>%
  group_by(line) %>%
  summarise(star = mean(stars))
w = which(!train$line   %in% xx$line)

star_mode = round(mean(train$stars))
my_tidy = tibble(
  line = w,
  stars = rep(star_mode, length(w)),
  word = rep("myWord", length(w)),
  count = rep(1, length(w))
)

tidy_train3 <- full_join(tidy_train2, my_tidy, by = c("line", "stars", "word", "count")) %>%
  arrange(line)

tidy_test2 <- tidy_test %>%
  inner_join(major, by = "word") %>%
  mutate(count = 1)
xx <- tidy_test2 %>%
  group_by(line) %>%
  summarise(star = mean(stars))
w = which(!test$line   %in% xx$line)


my_tidy = tibble(
  line = w,
  stars = rep(star_mode, length(w)),
  word = rep("myWord", length(w)),
  count = rep(1, length(w))
)

t2 = Sys.time()
tidy_test3 <- full_join(tidy_test2, my_tidy, by = c("line", "stars", "word", "count")) %>%
  arrange(line)
t3 = Sys.time()
print(t3-t2)


tidy_train3$line = tidy_train3$line + n1

tidy_train4 <- tidy_train3 %>%
  bind_tf_idf(word, line, count)
tidy_test4 <- tidy_test3 %>%
  bind_tf_idf(word, line, count)

tidy_all <- bind_rows(tidy_test4, tidy_train4)


t2 = Sys.time()
all_matrix <- tidy_all %>%
  cast_sparse(line, word, count)
t3 = Sys.time()
print(t3-t2)

starss = c(rep(0, n1), train$stars)

n2 = dim(all_matrix)[2]
all_matrix2 = cbind(all_matrix, starss)
test_index = c(1:n1)
train_matrix = all_matrix2[-test_index,]
test_matrix = all_matrix2[test_index,]
fit = glmnet(train_matrix[,1:n2], train_matrix[,n2+1])
cv <- cv.glmnet(train_matrix[,1:n2], train_matrix[,n2+1], nfolds=5)



# predict -----------------------------------------------------------------


pred <- predict(fit, test_matrix[, 1:n2], type="response", s=cv$lambda.min)


my_sh = c(1.5, 2.5, 3.5, 4.5)


pred_new = mypred(my_sh, pred)


result = tibble(Id = c(1:length(pred_new)),
                Prediction1 = pred_new)

write_csv(result, "/Users/Lyf/Desktop/result.csv")


write_csv(tidy_train3, "/Users/Lyf/Desktop/tidy_train3.csv")
write_csv(tidy_test3, "/Users/Lyf/Desktop/tidy_test3.csv")
