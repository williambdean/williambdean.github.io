---
tags: 
    - Python
    - Bayesian Statistics
comments: true
---

# Conjugate Priors

Bayesian statistics is a great way to think about data under the uncertainty of model parameters and I've found conjugate priors to be a good way to get started with some problems. 

## What is a Conjugate Prior?

A conjugate prior is a prior distribution that is in the same family as the posterior distribution. This is a mathematic convenience that makes it easier to calculate the posterior distribution, often just with a simple addition to the parameters of the prior distribution using the observed data. 

For instance, if we have a Bernoulli distribution with a single unknown success rate, a Beta prior on the success rate results in a posterior distribution that is also a Beta distribution. The posterior distribution is just the prior distribution with the addition of the number of successes and failures.

```python title="Get Posterior Distribution of Bernoulli Distribution with Beta Prior"" 
from conjugate.distributions import Beta
from conjugate.models import binomial_beta

N = 10
X = 4

prior = Beta(alpha=1, beta=1)
posterior: Beta = binomial_beta(n=N, x=X, beta_prior=prior)
```

Often times as well, the posterior predictive distribution in in a closed form distribution too. This provides new, alternative data sets. For instance, the posterior predictive distribution of a Bernoulli distribution with a Beta prior is a Beta-Binomial distribution.

```python title="Get Posterior Predictive Distribution of Bernoulli Distribution with Beta Prior""
from conjugate.distributions import BetaBinomial
from conjugate.models import binomial_beta_posterior_predictive

posterior_predictive: BetaBinomial = binomial_beta_posterior_predictive(
    n=N, 
    beta=posterior
) 
```

Having a closed form distribution for the posterior distribution and posterior predictive distribution can be useful to quickly assess the data, make decisions, and communicate more than just a parameter point estimate.


!!! Tip
    The **prior** predictive distribution is also a Beta-Binomial distribution. Just the posterior predictive with prior info. This provides the results we *expect* to see before we see the actual data.

Below is visualization of 10 trials, before and after we see the data.

![Binomial Model](../images/binomial-beta.png)


## Common Conjugate Models

Many common distributions like the Bernoulli, Poisson, or Normal distributions have conjugate models. They show up a lot in the single parameter distributions or where all but one parameter is known. 

Wikipedia provides table of common conjugate models [here](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions).

## Reducing Problem to a Conjugate Model

Though some data problems scream out these common distributions. For instance,

- Count data can be modeled with a Poisson distribution
- Binary data can be modeled with a Bernoulli distribution
- Number of successes / single outcomes in a fixed number of trials / attempts can be modeled with a Binomial distribution
- Sum of independent identical distributed variables can be modeled with a [Normal distribution](https://en.wikipedia.org/wiki/Central_limit_theorem)

If not, many questions on the data can be reduced to one of these common distributions. 

For instance, binning data into two groups can be modeled with a Bernoulli or Binomial distribution. This can be helpful in understanding the tail of a distribution. 

Knowing common [relationships between distributions](https://en.wikipedia.org/wiki/Relationships_among_probability_distributions) are useful to understand as well. 


## Why Use Conjugate Priors?

Conjugate priors can be a starting point due to their simplicity.

The availability of quantiles and moments can be useful to understand the data, even if the observed data falls outside of the posterior and posterior predictive distributions.

They can give a sense of fit or the lack of.

## Using Conjugate Priors in Different Settings

Since there is a closed for the posterior distribution, these models could also be implemented in SQL and moments could be calculated in SQL as well. This could be useful for large datasets that are too large to fit in memory and could be useful for a quick analysis backed with statistical theory.

Though these distributions are simple, they can be applied at a very granular level. For instance, a single user's click through rate could be modeled with a Bernoulli distribution. This could be useful to understand the uncertainty of a single user's click through rate. A quick win for user level personalization could be to show the user the content with the highest posterior predictive distribution. Or, some relevant [reward function](https://bayesiancomputationbook.com/markdown/chp_09.html#reward-functions-and-decisions) could be used to determine the best content to show the user.

## Summary

Conjugate priors are a great way to get started with Bayesian statistics. They provide a closed form posterior distribution and, often, posterior predictive distribution. This can be useful to understand the data and make decisions.

They are good for simple models and can be applied at a granular level.

If you're interested in trying out in python or want to see more examples, check out my [repo](https://github.com/wd60622/conjugate/) and [docs](https://wd60622.github.io/conjugate/) to use conjugate priors. 
