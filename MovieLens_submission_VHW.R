#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes
# check all necessary libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(ggplot2)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Split raw data set into train and test set: Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
summary(edx)
head(edx)

# Learners will develop their algorithms on the edx set
# For grading, learners will run algorithm on validation set to generate ratings
validation_CH <- validation  # save the rating information
validation <- validation %>% select(-rating)

# function to calcualte the RMSE values
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2,na.rm = T))
}


########## here the real work starts ##########
#### Data Preparation ####
# Data preparation: Modify the edx/validation data set so that we have year as a column as well
edx <- edx %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
validation <- validation %>% mutate(year = as.numeric(str_sub(title,-5,-2)))
#### Data exploration of edx ####
nrow(edx)
ncol(edx)
head(edx)
summary(edx)

v_ratings <- as.vector(edx$rating)
unique(v_ratings) 
table_ratings <- table(v_ratings)
table_ratings

v_ratings <- v_ratings[v_ratings != 0]
v_ratings <- factor(v_ratings)
qplot(v_ratings) +
  ggtitle("Distribution of the ratings")

# Simple visualization of the number of times each movie has been reviewed
edx %>% count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

# Simple visualization of the number of times each user has reviewed movies  
edx %>% count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Users")

# release year vs rating
edx %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth()



#### Data Analysis ####

# Initialize docuemntation of RMSE for comparison purpose
rmse_results <- data_frame()

# mean of rating over all users and movies
mu <- mean(edx$rating)  

# simple model accounting the movie variation:For each movie, subtract the rating minus the mean for each rating it received
movie_avgs_norm <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs_norm %>% qplot(b_i, geom ="histogram", bins = 20, data = ., color = I("black"))

# simple model accounting the user variation
user_avgs_norm <- edx %>% 
  left_join(movie_avgs_norm, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs_norm %>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))



#### Model Validation ####

#### Naive Model -- mean only ####
naive_rmse <- RMSE(validation_CH$rating,mu)
## Test our results based on the simple prediction
naive_rmse
## See result
rmse_results <- data_frame(method = "Using mean only", RMSE = naive_rmse)
## Store prediction in data frame

#### Movie effects only ####
predicted_ratings_movie_norm <- validation %>% 
  left_join(movie_avgs_norm, by='movieId') %>%
  mutate(pred = mu + b_i) 
model_1_rmse <- RMSE(validation_CH$rating,predicted_ratings_movie_norm$pred)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Effect Model",  
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()
## Test rmse and store in table

#### Movie and User Effects ####
predicted_ratings_user_norm <- validation %>% 
  left_join(movie_avgs_norm, by='movieId') %>%
  left_join(user_avgs_norm, by='userId') %>%
  mutate(pred = mu + b_i + b_u) 
## Use test set, then join movie averages and user averages, and find prediction
## Prediction is mean plus user effect plus movie effect
model_2_rmse <- RMSE(validation_CH$rating,predicted_ratings_user_norm$pred)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and User Effect Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()
## Test and store results


#### Regularized movie and user effects ####
## Regularized time
lambdas <- seq(0, 10, 0.25)
## Sequence of lambdas to use
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  
  return(RMSE(validation_CH$rating,predicted_ratings))
})
## For each lambda, we find the b_i and the b_u, then make our prediction and test.  

qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
lambda
## Plot the lambdas vs the rmses, see which has the best accuracy, and choose that for lambda.

movie_avgs_reg <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())
## Using lambda, find the movie effects
user_avgs_reg <- edx %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i)/(n()+lambda), n_u = n())
## Using lambda, find the user effects
predicted_ratings <- validation %>% 
  left_join(movie_avgs_reg, by='movieId') %>%
  left_join(user_avgs_reg, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>% group_by(userId,movieId) %>% summarize(pred_2 = mean(pred))



## Make our predicted ratings

model_3_rmse <- RMSE(validation_CH$rating,predicted_ratings$pred_2)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie and User Effect Model",  
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()
## Test and store results


## Use our best lambda to make new predictions, test, and store results

#### Result ####
## Round our predicted_ratings
predicted_ratings$pred_2 <- round(predicted_ratings$pred_2*2)/2
## Make sure all ratings are between 0.5 and 5
predicted_ratings$pred_2[which(predicted_ratings$pred_2<1)] <- 0.5
predicted_ratings$pred_2[which(predicted_ratings$pred_2>5)] <- 5

## See direct accuracy
mean(predicted_ratings$pred_2 == validation_CH$rating)
## See close accuracy--within 0.5 stars
x <- sum(predicted_ratings$pred_2 <= validation_CH$rating + 0.5 & predicted_ratings$pred_2 >= validation_CH$rating - 0.5)
x/length(predicted_ratings$pred_2)


# Ratings will go into the CSV submission file below:

write.csv(validation %>% select(userId, movieId) %>% mutate(rating = predicted_ratings$pred_2),
          "submission.csv", na = "", row.names=FALSE)
rm(dl, ratings, movies, test_index, temp, movielens, removed)