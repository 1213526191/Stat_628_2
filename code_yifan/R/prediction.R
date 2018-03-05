library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)
library(glmnet) 

# mode fun ----------------------------------------------------------------

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# red function ------------------------------------------------------------

mypred = function(sh, pred_ori){
  pred_new = rep(0, length(pred_ori))
  pred_new[which(pred_ori<sh[1])] = 1
  pred_new[which(pred_ori>=sh[1] & pred_ori<sh[2])] = 2
  pred_new[which(pred_ori>=sh[2] & pred_ori<sh[3])] = 3
  pred_new[which(pred_ori>=sh[3] & pred_ori<sh[4])] = 4
  pred_new[which(pred_ori>=sh[4])] = 5
  return(pred_new)
}

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



# fit model ---------------------------------------------------------------


t1 = Sys.time()
tidy_train <- train %>%
  unnest_tokens(word, text)
t2 = Sys.time()
print(t2-t1)

tidy_train <- tidy_train %>%
  inner_join(major, by = "word") %>%
  mutate(count = 1)
xx <- tidy_train %>%
  group_by(line) %>%
  summarise(star = mean(stars))
w = which(!train$line   %in% xx$line)

star_mode = Mode(train$stars)
my_tidy = tibble(
  line = w,
  stars = rep(star_mode, length(w)),
  word = rep("myWord", length(w)),
  count = rep(1, length(w))
)

tidy_train2 <- full_join(tidy_train, my_tidy, by = c("line", "stars", "word", "count")) %>%
  arrange(line)

# train_dtm <- tidy_train %>%
#   cast_dtm(line, word, count)

train_matrix <- tidy_train2 %>%
  cast_sparse(line, word, count)
n2 = dim(train_matrix)[2]
train_matrix2 = cbind(all_matrix, train$stars)
fit = glmnet(train_matrix2[,1:n2], train_matrix2[,n2+1])
cv <- cv.glmnet(train_matrix2[,1:n2], train_matrix2[,n2+1], nfolds=5)



# predict -----------------------------------------------------------------

t2 = Sys.time()
tidy_test <- test %>%
  unnest_tokens(word, text)
t3 = Sys.time()
print(t3-t2)

tidy_test <- tidy_test %>%
  inner_join(major, by = "word") %>%
  mutate(count = 1)
xx <- tidy_test %>%
  group_by(line) %>%
  summarise(star = mean(stars))
w = which(!test$line   %in% xx$line)

star_mode = Mode(train$stars)
my_tidy = tibble(
  line = w,
  stars = rep(star_mode, length(w)),
  word = rep("myWord", length(w)),
  count = rep(1, length(w))
)

t2 = Sys.time()
tidy_test2 <- full_join(tidy_test, my_tidy, by = c("line", "stars", "word", "count")) %>%
  arrange(line)
t3 = Sys.time()
print(t3-t2)

t2 = Sys.time()
test_matrix <- tidy_test2 %>%
  cast_sparse(line, word, count)
t3 = Sys.time()
print(t3-t2)



pred <- predict(fit, test_matrix, type="response", s=cv$lambda.min)
pred_new = mypred(my_sh, pred)


result = tibble(Id = c(1:length(pred_new)),
                Prediction1 = pred_new)

write_csv(result, "/Users/Lyf/Desktop/result.csv")
