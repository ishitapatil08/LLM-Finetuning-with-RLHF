# LLM-Finetuning-with-RLHF
Reinforcement Learning from Human Feedback (RLHF) is a method that integrates human feedback and reinforcement learning into LLMs, enhancing their alignment with human objectives and improving their efficiency.

RLHF has shown significant promise in making LLMs safer and more helpful. It was used for the first time for creating InstructGPT, a finetuned version of GPT3 for following instructions, and it’s used nowadays in the last OpenAI models ChatGPT (GPT-3.5-turbo) and GPT-4.

RLHF leverages human-curated rankings that act as a signal to the model, directing it to favor specific outputs over others, thereby encouraging the production of more reliable, secure responses that align with human expectations. All of this is done with the help of a reinforcement learning algorithm, namely PPO, that optimizes the underlying LLM model, leveraging the human-curated rankings.
