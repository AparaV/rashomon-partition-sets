# install.packages("ppmSuite")

library(ppmSuite)
library(data.table)


set.seed(0)

# Prior params
M <- 1
mu0 <- 0
s20 <- 4
v2 <- 0.5
simParams <- c(mu0, s20, v2, 1, 2, 0.1, 1)

# Sim parameters
ndraws <- 1100
nburn <- 100

true_best_effect <- 4.5
true_best_features <- matrix(c(4, 1, 4, 2, 4, 3, 4, 4), ncol=2, byrow=TRUE)
colnames(true_best_features) <- c("X0", "X1")
true_best_features_dt <- as.data.table(true_best_features)

min_dosage_best_effect_feature <- matrix(c(4, 1), nrow=1)
colnames(min_dosage_best_effect_feature) <- c("X0", "X1")
min_dosage_best_effect_feature_dt <- as.data.table(min_dosage_best_effect_feature)

# Data parameters
P <- 2
num_sims <- 100
sample_sizes <- c(10, 20, 50, 100, 500, 1000)


# Output dataframe
output_df <- data.frame(
    n_per_pol=integer(),
    sim_num=integer(),
    sample_idx=integer(),
    neg_log_posterior=numeric(),
    nclusters=integer(),
    MSE=numeric(),
    IOU=numeric(),
    min_dosage=integer(),
    best_pol_diff=numeric()
)
    

# Read data

for (n_per_pol in sample_sizes) {
    
    
    print("======")
    print(paste("n_per_pol =", n_per_pol))
    
    for (sim_num in c(0:(num_sims-1))) {
        
        verbose <- FALSE
        if ((sim_num+1) %% 25 == 0) {
            print(paste("    Simulation", sim_num))
            verbose <- TRUE
        }
    
        data_fname <- paste0("../../Data/worst_case_sims/sim_data_", n_per_pol, "_", sim_num, ".csv")
        data <- read.csv(data_fname)
    
        X <- data[, c("X0", "X1")]
        num_data <- nrow(X)
        X2 <- X^2
        y <- data$y
    
        res <- gaussian_ppmx(
            y,
            X,
            Xpred = NULL,
            simParms = simParams,
            draws=ndraws, burn=nburn,
            verbose=verbose)
    
    
        fitted_vals <- res$fitted.values
        
        # Compute MSE
        errs <- sweep(fitted_vals, MARGIN=2, y, FUN="-")
        sqrd_errs <- errs^2
        mse <- rowSums(sqrd_errs) / num_data
        
    
        clusters <- res$Si
        nclus <- res$nclus
        log_like <- log(res$like)
        log_like_iters <- rowSums(log_like)
    
        niters <- ndraws - nburn
    
        for (i in c(1:niters)) {
            
            nclusters_i <- nclus[i]
            cluster_i <- clusters[i, ]
            cluster_freq <- as.vector(table(cluster_i))
            
            # Compute best cluster
            y_fitted_i <- fitted_vals[i, ]
            y_clus_means <- rowsum(y_fitted_i, group = cluster_i) / cluster_freq
            best_cluster_i <- which.max(y_clus_means)
            best_feature_idx_i <- which(cluster_i == best_cluster_i)
            best_features_i <- unique(X[best_feature_idx_i, ])
            best_features_dt_i <- as.data.table(best_features_i)
            
            # Best cluster diff
            best_cluster_diff_i <- true_best_effect - max(y_clus_means)
            
            # Min dosage best cluster
            min_dosage_intersection_i <- nrow(fintersect(best_features_dt_i, min_dosage_best_effect_feature_dt))
            min_dosage_included_i <- min_dosage_intersection_i > 0
            
            # Intersection-over-union
            intersection_i <- nrow(fintersect(best_features_dt_i, true_best_features_dt))
            union_size_i <- nrow(funion(best_features_dt_i, true_best_features_dt))
            iou_i <- intersection_i / union_size_i
            
            # Compute posterior
            log_prior <- 0
    
            log_clus <- log(M) + 0.5 * nclusters_i * (nclusters_i - 1)
            log_prior <- log_prior + log_clus
    
            sumX <- rowsum(X, group = cluster_i)
            sumX2 <- rowsum(X2, group = cluster_i)
    
            for (k in c(1:nclusters_i)) {
                nk <- cluster_freq[k]
                for (p in c(1:P)) {
                    s2s <- 1/((nk/v2) + (1/s20))
                    mus <- s2s*((1/v2)*sumX[k, p] + (1/s20)*mu0)
    
                    ld1 <- -0.5*nk*log(2*pi*v2) - 0.5*(1/v2)*sumX2[k, p]
                    ld2 <- dnorm(mu0, 0, sqrt(s20), log=TRUE)
                    ld3 <- dnorm(mus, 0, sqrt(s2s), log=TRUE)
    
                    log_prior <- log_prior + ld1 + ld2 - ld3
                }
            }
    
            log_post_i <- log_like_iters[1] + log_prior
    
            # Write to output data frame        
            result_row_i <- c(
                n_per_pol,
                sim_num,
                i,
                -log_post_i,
                nclusters_i,
                mse[i],
                iou_i,
                min_dosage_included_i,
                best_cluster_diff_i
            )
            
            output_df <- rbind(output_df, result_row_i)
        }
    
    }
}

colnames(output_df) <- c("n_per_pol", "sim_num", "sample_idx", "neg_log_posterior", "nclusters", "MSE", "IOU", "min_dosage", "best_pol_diff")

output_df_name <- paste0("../../Results/worst_case/worst_case_ppmx.csv")

write.csv(output_df, output_df_name, quote=FALSE)

print(paste("Wrote results to:", output_df_name))
print("======")
