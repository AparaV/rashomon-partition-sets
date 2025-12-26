# install.packages("ppmSuite")

library(ppmSuite)


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

# Data parameters
P <- 4
num_sims <- 100


# Read data

for (sim_num in c(0:(num_sims-1))) {
    
    print(paste("Simulation", sim_num))
    
    data_fname <- paste0("../../Data/reff_sims/sim_data_30_", sim_num, ".csv")
    data <- read.csv(data_fname)
    
    X <- data[, c("X0", "X1", "X2", "X3")]
    X2 <- X^2
    y <- data$y
    
    res <- gaussian_ppmx(
        y,
        X,
        Xpred = NULL,
        simParms = simParams,
        draws=ndraws, burn=nburn,
        verbose=TRUE)
    
    
    fitted_vals <- res$fitted.values
    
    clusters <- res$Si
    nclus <- res$nclus
    log_like <- log(res$like)
    log_like_iters <- rowSums(log_like)
    
    
    log_posteriors <- c()
    
    niters <- ndraws - nburn
    
    for (i in c(1:niters)) {
        nclusters_i <- nclus[i]
        cluster_i <- clusters[i, ]
        cluster_freq <- as.vector(table(cluster_i))
        
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
        
        log_posteriors <- c(log_posteriors, log_post_i)
    }
    
    
    fitted_vals_fname <- paste0("../../Results/reff/ppmx/fitted_vals_", sim_num, ".csv")
    nclusters_vals_fname <- paste0("../../Results/reff/ppmx/nclusters_", sim_num, ".csv")
    posteriors_fname <- paste0("../../Results/reff/ppmx/posteriors_", sim_num, ".csv")
    
    write.csv2(fitted_vals, fitted_vals_fname)
    write.csv2(nclus, nclusters_vals_fname)
    write.csv2(log_posteriors, posteriors_fname)
    
    print(paste("Wrote results to:", fitted_vals_fname, nclusters_vals_fname, "and", posteriors_fname))
    print("====")

}