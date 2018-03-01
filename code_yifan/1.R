library(tidyverse)
library(tidytext)
library(Matrix)
library(topicmodels)


# read data ---------------------------------------------------------------

data_ori = read_csv("data.csv")

data_df = data_frame(line = 1:nrow(data_ori),
                     text = data_ori$text, 
                     stars = data_ori$stars)
n1 = length(data_df$line)

# CV ----------------------------------------------------------------------

cv_n = 5
cv_df = tibble(index = c(1:cv_n))
set.seed(615)
cv_index = sample_n(cv_df, nrow(data_df), replace = T)


# not dict ----------------------------------------------------------------

data("stop_words")
add_not = function(x){
  return(paste("not_", x[1], sep = ""))
}
my_stop <- stop_words 
my_stop$word = apply(my_stop, 1, add_not)


# mode fun ----------------------------------------------------------------

Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# split data --------------------------------------------------------------

var_sh = 1/300
mse = numeric(cv_n)
for(ii in 1:cv_n){
  split <- which(cv_index$index != ii)
  train <- data_df[split,]
  test <- data_df[-split,]
  tidy_train <- train %>%
    unnest_tokens(word, text)
  tidy_test <- test %>%
    unnest_tokens(word, text)
  
  
  # delete stop word --------------------------------------------------------
  
  
  
  # tidy_train <- tidy_train %>%
  #   anti_join(stop_words, by = "word") %>%
  #   anti_join(my_stop, by = "word")
  
  
  # delete rear word --------------------------------------------------------
  
  # tidy_train %>%
  #   count(word, sort = T) %>%
  #   filter(n > 1000) %>%
  #   mutate(word = reorder(word, n)) 
  
  major <- tidy_train %>%
    count(word, sort = T) %>%
    filter(n > 100)
  
  tidy_train <- tidy_train %>%
    inner_join(major, by = "word")
  
  
  # delete useless word -----------------------------------------------------
  
  word_var <- data_frame(word = major$word, var = rep(0, nrow(major)))
  tidy_train_count <- tidy_train %>%
    count(stars)
  for(i in 1:nrow(word_var)){
    this_word = word_var$word[i]
    x <- tidy_train %>%
      filter(word == this_word) %>%
      count(stars) 
    my_star = tibble(stars = c(1:5), nn = rep(0, 5))
    x2 <- rbind(x, my_star) %>%
      group_by(stars) %>%
      summarise(nn = max(nn)) %>%
      mutate(xx = nn/tidy_train_count$nn) %>%
      mutate(xx = xx/sum(xx))
    word_var$var[i] = var(x2$xx)
    # cat(i)
  }
  
  word_var <- word_var %>%
    arrange(desc(var))
  
  
  # for(i in 1:10){
  #   this_word = word_var$word[i]
  #   x <- tidy_test %>%
  #     filter(word == this_word) %>%
  #     count(stars) %>%
  #     mutate(xx = n/sum(n))
  #   ggplot(data = x) +
  #     geom_bar(aes(x = stars, y = xx), stat = 'identity') +
  #     ggtitle(this_word)
  # }
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
  
  
  
  # inspect(all_sp[20, 1:20])
  dtm.to.Matrix <- function(dtm)
  {
    m <- Matrix(0, nrow = dtm$nrow, ncol = dtm$ncol, sparse = TRUE)
    for (index in 1:length(dtm$i)){
      m[dtm$i[index], dtm$j[index]] <- dtm$v[index]
    }
    return(m)
  }
  all_matrix = dtm.to.Matrix(all_sp)
  
  all_matrix2 = cbind(all_matrix, data_df$stars)
  
  
  # fit model ---------------------------------------------------------------
  
  split2 = split
  train2 <- all_matrix2[split2,]
  test2 <- all_matrix2[-split2,]
  true_value = data_df$stars[-split2]
  library(glmnet) 
  
  
  fit = glmnet(train2[,1:n2], train2[,n2+1])
  
  cv <- cv.glmnet(train2[,1:n2], train2[,n2+1],nfolds=5)
  pred <- predict(fit, test2[,1:n2],type="response", s=cv$lambda.min)
  
  pred2 = round(pred)
  pred2[which(pred2>5)]=5
  pred2[which(pred2<0)]=0
  mse[ii] = sum((round(pred)-true_value)^2)
  print(ii)
}
sum(mse)/n1



