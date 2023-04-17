# Calculates prediction intervals of each nodules in the dataset
# Inputs:
#   ProbMatrix: Each row is a probability vector of each response class for a given annotation
#   Labels: vector of names of response classes (in same order as columns of the ProbMatrix)
# Return:
#   pred50 has 50% prediction intervals
#   pred80 has 80% prediction intervals

CategoryPredIntervalNodules = function(ProbMatrix, labels) { 
  test_annotation <- read.csv("../Data/Meta/meta_annotation_info_test.csv", header = TRUE)
  
  nannotations = nrow(ProbMatrix)
  nodule_id = test_annotation$Nodule_id
  n_nodules = length(unique(nodule_id))
  pred50 = rep(NA,n_nodules); pred80 = rep(NA,n_nodules)
  
  nodule_count = 1
  c = 1
  num_annotations = 0
  psum = 0
  curr_nodule_id = nodule_id[1]
  for(i in 1:nannotations){
    if (i != nannotations) {
      c = c + 1
      curr_nodule_id = nodule_id[c]
      num_annotations = num_annotations + 1
      psum = psum + ProbMatrix[i,]
    }
    if (nodule_id[i] != curr_nodule_id) {
      psum_avg = psum / num_annotations

      ip = order(psum_avg,decreasing=T)
      pOrdered = psum_avg[ip] # decreasing order
      labelsOrdered = labels[ip] # decreasing order
      G = cumsum(pOrdered) # cumulative sum from largest
      k1 = min(which(G>=0.5)) # level1= 0.5
      k2 = min(which(G>=0.8)) # level2= 0.8
      pred1 = labelsOrdered[1:k1]; pred2 = labelsOrdered[1:k2]
      pred50[nodule_count] = paste(pred1,collapse="")
      pred80[nodule_count] = paste(pred2,collapse="") 
      
      psum = 0
      num_annotations = 0
      nodule_count = nodule_count + 1
    }
  }
  list(pred50=pred50, pred80=pred80)
}

# Purpose: Produce coverage rates of prediction intervals by response class
# Input:
#   pred_intervals: One of the two objects (pred50 or pred80) outputted by CategoryPredIntervalNodules()
#   is_50_percent: TRUE if pred50, FALSE if pred80
# Output:
#   A list containing the coverage rates of each class in the order: true, false, ambiguous, no consensus

getCoverageRateByClassNodule = function(pred_intervals, level) {
  test_nodule <- read.csv("../Data/Meta/meta_nodule_info_test.csv", header = TRUE)
  observed = test_nodule$Is_cancer
  observed = as.character(observed)
  
  n = nrow(test_nodule)
  n_true = nrow(subset(test_nodule, Is_cancer == "True"))
  n_false = nrow(subset(test_nodule, Is_cancer == "False"))
  n_ambiguous = nrow(subset(test_nodule, Is_cancer == "Ambiguous"))
  n_no_consensus = nrow(subset(test_nodule, Is_cancer == "No_consensus"))
  
  if (level == 50) {
    pred_intervals_confidence = pred_intervals$pred50
  } else {
    pred_intervals_confidence = pred_intervals$pred80
  }
  
  coverage_rate_true = 0
  coverage_rate_false = 0
  coverage_rate_ambiguous = 0
  coverage_rate_no_consensus = 0
  
  for (i in 1:n) {
    if(!grepl(observed[i], pred_intervals_confidence[i], fixed=TRUE)){
      next
    }
    if(observed[i] == "True") {
      coverage_rate_true = coverage_rate_true + 1
    } else if(observed[i] == "False") {
      coverage_rate_false = coverage_rate_false + 1
    } else if(observed[i] == "Ambiguous") {
      coverage_rate_ambiguous = coverage_rate_ambiguous + 1
    } else {
      coverage_rate_no_consensus = coverage_rate_no_consensus + 1
    }
  }
  
  coverage_rate_true = coverage_rate_true / n_true
  coverage_rate_false = coverage_rate_false / n_false
  coverage_rate_ambiguous = coverage_rate_ambiguous / n_ambiguous
  coverage_rate_no_consensus = coverage_rate_no_consensus / n_no_consensus
  
  list(coverage_rate_true, coverage_rate_false, coverage_rate_ambiguous, coverage_rate_no_consensus)
}

# Purpose: Produce overall coverage rates of prediction interval
# Input:
#   pred_intervals: One of the two objects (pred50 or pred80) outputted by CategoryPredIntervalNodules()
# Output:
#   A list containing the coverage rate

getOverallCoverageRateNodule = function(pred_intervals, level) {
  test_nodule <- read.csv("../Data/Meta/meta_nodule_info_test.csv", header = TRUE)
  observed = test_nodule[c("Is_cancer")]
  
  if (level == 50) {
    pred_intervals_confidence = pred_intervals$pred50
  } else {
    pred_intervals_confidence = pred_intervals$pred80
  }
  
  n = length(pred_intervals_confidence)
  
  coverage_rate = 0
  for (i in 1:(n-1)){
    if(grepl(observed[i, ], pred_intervals_confidence[i], fixed=TRUE)) {
      coverage_rate = coverage_rate + 1
    }
  }
  coverage_rate = coverage_rate / n
  list(coverage_rate)
}



