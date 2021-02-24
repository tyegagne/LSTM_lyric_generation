Text generation with LSTM
================

## Implementing character-level LSTM text models for lyrical starting point generation

Here we’ll demonstrate a proof-of-concept regarding lyric generation via
lyric seed after training a LSTM on a corpus of folk lyrics. The first
thing we start with is about 100,000 lyrics pulled from a folk song
database. In the future this can be tailored ot americana, appalacia,
bluegrass, etc. The language model ideally will learn a model of the
writing style on which it was trained.

## Preparing the data

Let’s start by downloading the corpus and converting it to lowercase:

``` r
library(tidyverse)
library(keras)
library(stringr)

path <- '/Users/tyler.gagne/LSTM_lyric_generation/data/folk_lyrics.txt'
path <- read_file(path)

# generate lowercase and remove irrelevant punctuation
text <- tolower(path)
text <- gsub('[[:punct:] ]+',' ',text)

# subset a million characters for this PoC (saves compute time)
text <- substr(text,1,1000000)
str(text)
```

    ##  chr " x \n 1 rise and fall like the tide\nmy hand goes with your chest\nsteady now moon will pull\na slow and even b"| __truncated__

``` r
cat("Corpus length:", nchar(text), "\n")
```

    ## Corpus length: 1000000

Here we extract overlapping sequences of characters of length 30, encode
it, and put it in to a 3D array of x of shape (sequences, maxlen, unique
characters)

We also prepare and array ‘y’ containing the corresponding targets,
i.e. the encoded characters that come after each extracted sequence.

``` r
maxlen <- 30  # Length of extracted character sequences
step <- 3  # We sample a new sequence every `step` characters
  
text_indexes <- seq(1, nchar(text) - maxlen, by = step)
# This holds our extracted sequences
sentences <- str_sub(text, text_indexes, text_indexes + maxlen - 1)
# This holds the targets (the follow-up characters)
next_chars <- str_sub(text, text_indexes + maxlen, text_indexes + maxlen)
cat("Number of sequences: ", length(sentences), "\n")
```

    ## Number of sequences:  333324

``` r
# List of unique characters in the corpus
chars <- unique(sort(strsplit(text, "")[[1]]))
cat("Unique characters:", length(chars), "\n")
```

    ## Unique characters: 63

``` r
# Dictionary mapping unique characters to their index in `chars`
char_indices <- 1:length(chars) 
names(char_indices) <- chars
# Next, one-hot encode the characters into binary arrays.
cat("Vectorization...\n") 
```

    ## Vectorization...

``` r
x <- array(0L, dim = c(length(sentences), maxlen, length(chars)))
y <- array(0L, dim = c(length(sentences), length(chars)))
for (i in 1:length(sentences)) {
  sentence <- strsplit(sentences[[i]], "")[[1]]
  for (t in 1:length(sentence)) {
    char <- sentence[[t]]
    x[i, t, char_indices[[char]]] <- 1
  }
  next_char <- next_chars[[i]]
  y[i, char_indices[[next_char]]] <- 1
}
```

## Building the network

This network is a single LSTM layer followed by a dense classifier and
softmax over all possible characters. But note that recurrent neural
networks aren’t the only way to do sequence data generation; 1D convnets
also have proven extremely successful at this task in recent times.

``` r
model <- keras_model_sequential() %>% 
  layer_lstm(units = 128, input_shape = c(maxlen, length(chars))) %>% 
  layer_dense(units = length(chars), activation = "softmax")
```

Since our targets are one-hot encoded, we will use
`categorical_crossentropy` as the loss to train the model:

``` r
optimizer <- optimizer_rmsprop(lr = 0.01)
model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = optimizer
)   
```

## Training the language model and sampling from it

Given a trained model and a seed text snippet, we generate new text by
repeatedly:

  - 1)  Drawing from the model a probability distribution over the next
        character given the text available so far

  - 2)  Reweighting the distribution to a certain “temperature”

  - 3)  Sampling the next character at random according to the
        reweighted distribution

  - 4)  Adding the new character at the end of the available text

This is the code we use to reweight the original probability
distribution coming out of the model, and draw a character index from it
(the “sampling function”):

``` r
sample_next_char <- function(preds, temperature = 1.0) {
  preds <- as.numeric(preds)
  preds <- log(preds) / temperature
  exp_preds <- exp(preds)
  preds <- exp_preds / sum(exp_preds)
  which.max(t(rmultinom(1, 1, preds)))
}
```

Finally, the following loop repeatedly trains and generates text. You
begin generating text using a range of different temperatures after
every epoch. This allows you to see how the generated text evolves as
the model begins to converge, as well as the impact of temperature in
the sampling strategy.

``` r
for (epoch in 1:1) {
  
  cat("epoch", epoch, "\n")
  
  # Fit the model for 1 epoch on the available training data
  model %>% fit(x, y, batch_size = 128, epochs = 1, verbose = 0) 
  
  # Select a text seed at random
  start_index <- sample(1:(nchar(text) - maxlen - 1), 1)  
  seed_text <- str_sub(text, start_index, start_index + maxlen - 1)

  cat("--- Generating with seed:", seed_text, "\n\n")
  
  for (temperature in c(0.2, 0.5, 1.0, 1.2)) {
    
    cat("------ temperature:", temperature, "\n")
    cat(seed_text, "\n")
    
    generated_text <- seed_text
    
     # We generate 400 characters
    for (i in 1:400) {
      
      sampled <- array(0, dim = c(1, maxlen, length(chars)))
      generated_chars <- strsplit(generated_text, "")[[1]]
      for (t in 1:length(generated_chars)) {
        char <- generated_chars[[t]]
        sampled[1, t, char_indices[[char]]] <- 1
      }
        
      preds <- model %>% predict(sampled, verbose = 0)
      next_index <- sample_next_char(preds[1,], temperature)
      next_char <- chars[[next_index]]
      
      generated_text <- paste0(generated_text, next_char)
      generated_text <- substring(generated_text, 2)
      
      cat(next_char)
    }
    cat("\n\n")
  }
}
```

    ## epoch 1 
    ## --- Generating with seed:  se bet 
    ## a bejr bet
    ## mg vletlen 
    ## 
    ## ------ temperature: 0.2 
    ##  se bet 
    ## a bejr bet
    ## mg vletlen 
    ##  
    ## szen 
    ## szen 
    ## s megy szem 
    ## szem 
    ## trimba 
    ## szen 
    ## egy szem 
    ## sit 
    ## s mind 
    ## sta 
    ## egy szrbem 
    ## szen 
    ## a szen 
    ## szes 
    ## szel 
    ## szenk 
    ## szen 
    ## szempen 
    ## szem 
    ## s mind 
    ## s mind like to the sand the storm
    ## the store the stord it s a stoly
    ## when it s all the storm the want the starth the stord the strone
    ## we were the sand to the sand the sand
    ## and you we starth the want of the sand
    ## and it s a store with the storm
    ## and you lo
    ## 
    ## ------ temperature: 0.5 
    ##  se bet 
    ## a bejr bet
    ## mg vletlen 
    ##  
    ##  ki szpanna 
    ## hilpaz az egy szan 
    ## koran 
    ## sze 
    ## hegy sz trsping hangan 
    ## was a paben so back you love you so money
    ## and they were this the bost with the sunney
    ## when a make the sound on the way
    ## in the cares on the sure
    ## it s all the strange
    ## our live the right
    ## but we re the stard a thind have
    ## the were in the rands rays of the want and the stall the foot
    ## it s no will be the strone
    ## it s we was and the mor
    ## 
    ## ------ temperature: 1 
    ##  se bet 
    ## a bejr bet
    ## mg vletlen 
    ## em 
    ## namsza 
    ## ja mincs estttmimbl oona
    ## k jak h pem allbs 
    ## snors alg arnyml fomuri falal hhi pumnora rue werse
    ## they were your breed um
    ## the frices 
    ## the want are waty
    ## arthand aline somewas a lond
    ## there
    ## i with me
    ## it was eyows the stars
    ## take ittester reyoob your sky
    ## what you goon  rile
    ## when we re the s are dack the sourte choristarrout
    ## when hat it ricking ownte right
    ## and with we sand
    ## and hold it clack th
    ## 
    ## ------ temperature: 1.2 
    ##  se bet 
    ## a bejr bet
    ## mg vletlen 
    ##  veniksrhenem megden vel ozserti rihmel
    ## tod hisalistrgkedefilivohidad
    ## kitr lesves tom r
    ## sohnva bimern
    ## ylight lives into lied oh serowaters
    ## i reind a bried
    ## welt your sould his 
    ## your grice as fol 
    ## kisso 
    ## i1 shhir eativi som 
    ## vglaeszenki 
    ## vrbenrbak 
    ## fhmaimhntrtsih 
    ## digarheg dics cu i burach wills brook or ssain 
    ## that let t baked your offer away
    ## 
    ## than he god
    ## the epleached my and bolls a rinder want
    ## ou

We can see here in this early iteration that we initially have
relatively repetitive and predictable text, though local structure is
relatively realisitic given the amount of data we did start with.

Given enough epochs and increasing ‘temperature’ we may see creative
output that is almost novel, or a fantastic hook to leverage as lyrical
starting point. Perhaps integrated in to a melody generation type neural
network.

With more data, more compute tume, we can generate strings and samples
that are more coherent. This can be acheived with future data aquisition
and compute.

References:

Generative deep learning networks in R
<https://github.com/jjallaire/deep-learning-with-r-notebooks>

Melodic style transfer approaches <https://github.com/openai/jukebox>
