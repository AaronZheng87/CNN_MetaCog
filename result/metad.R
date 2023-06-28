library(tidyverse)
df <- read_csv("/Users/zhengyuanrui/Downloads/exptest.csv")

df2 <- df %>% 
  filter(trial_type%in%c("image-keyboard-response", "html-keyboard-response"))

df_exp <- df2  %>% 
  filter(exp_type %in% c("practice", "formal")) 
df_exp_nof <- df_exp %>% 
  filter(is.na(feedback))

df_formal <- df_exp_nof %>% 
  filter(exp_type == "formal")



df_formal <- df_formal[,6:ncol(df_formal)]



df_formal_trials <- df_formal %>% 
  filter(stimulus != '<div style="font-size:60px; color: white">+</div>')

df_formal_id <- df_formal_trials %>% 
  mutate(trials_id = rep(1:(nrow(df_formal_trials)/3), each = 3))

df_stim <-  df_formal_id %>% 
  filter(grepl(".png", stimulus))

df_type1 <- df_formal_id %>%
  filter(stimulus == "<p> </p>")

df_type2 <- df_formal_id %>%
  filter(stimulus == "<p style =' color : white'>你认为刚才你的判断是?</p>")

response_incorrect_img <- df_type1 %>% 
  filter(response == "null") %>% 
  pull(trials_id)

response_incorrect_confidence <- df_type2 %>% 
  filter(response == "null") %>% 
  pull(trials_id)

incor_id <- unique(append(response_incorrect_img, response_incorrect_confidence))


cor_type1 <- df_type1 %>% 
  filter(!(trials_id %in% incor_id)) %>% 
  select(acc, correct_response, decision, rt, trials_id)

cor_stim <- df_stim %>% 
  filter(!(trials_id %in% incor_id)) %>% 
  select(stimulus, trials_id)

cor_type2 <- df_type2 %>% 
  filter(!(trials_id %in% incor_id)) %>% 
  select(decision_2, trials_id)

df_stim_type1 <- left_join(cor_stim, cor_type1, by = "trials_id")
final <- left_join(df_stim_type1, cor_type2, by = "trials_id")

final_metad <- final %>% 
  mutate(stim = if_else(correct_response == "Good", 1, 0), 
         response = if_else(decision == "好人", 1, 0), 
         rating = if_else(decision_2 == "正确", 2, 1))

final_metad %>% 
  group_by(acc) %>% 
  summarise(n = n())

write_csv(final_metad, "sub1_metad.csv")
getwd()



df_cnn <- read_csv("/Users/zhengyuanrui/CNN_Moral-MetaCog/result/reuslt_cnn.csv")


df_cnn %>% 
  group_by(acc, confidence) %>% 
  summarise(n = n()
  )
library(metaSDT)
?metaSDT
?fit_meta_d_MLE()
