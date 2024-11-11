import numpy as np
from scipy.stats import norm

def select_article(user_preferences, company_values, alpha, w1, w2):
    num_articles = len(user_preferences)
    
    # Explore new articles with probability alpha
    if np.random.rand() < alpha:
        return np.random.randint(num_articles)
    
    # Exploit known preferences
    else:
        scores = w1 * user_preferences + w2 * company_values
        return np.argmax(scores)

def update(user_preferences, company_values, cumulative_rewards, num_selections, article_index, reward):
    cumulative_rewards[article_index] += reward
    num_selections[article_index] += 1
    
    # Update user preference estimate using Bayesian update
    user_preferences[article_index] = (user_preferences[article_index] * (num_selections[article_index] - 1) + reward) / num_selections[article_index]
    
    return user_preferences, company_values, cumulative_rewards, num_selections

def get_reward(user_preferences, company_values, article_index, w1, w2):
    user_satisfaction = norm.pdf(user_preferences[article_index], 0, 1)
    company_value = company_values[article_index]
    return w1 * user_satisfaction + w2 * company_value

def run_simulation(num_articles, num_iterations, alpha, w1, w2):
    user_preferences = np.random.normal(0, 1, num_articles)
    company_values = np.random.uniform(-1, 1, num_articles)
    cumulative_rewards = np.zeros(num_articles)
    num_selections = np.zeros(num_articles)
    
    total_reward = 0
    for _ in range(num_iterations):
        article_index = select_article(user_preferences, company_values, alpha, w1, w2)
        reward = get_reward(user_preferences, company_values, article_index, w1, w2)
        user_preferences, company_values, cumulative_rewards, num_selections = update(user_preferences, company_values, cumulative_rewards, num_selections, article_index, reward)
        total_reward += reward
    
    return total_reward

num_articles = 100
num_iterations = 1000
alpha = 0.5
w1 = 0.6
w2 = 0.4

total_reward = run_simulation(num_articles, num_iterations, alpha, w1, w2)
print(f"Total reward: {total_reward:.2f}")