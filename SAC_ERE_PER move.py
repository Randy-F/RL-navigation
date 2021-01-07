import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
import time
from drone_rgb_env import drone_env_heightcontrol
from CsvUtil import CsvUtil


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.log_std_linear = nn.Linear(hidden_size, action_size)
        
        self.cnn_layers = nn.Sequential(
            # 定义2D卷积层
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 定义另一个2D卷积层
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.linear_layers1 = nn.Sequential(
            nn.Linear(16 * 64 * 1, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

        self.linear_layers2 = nn.Sequential(
            nn.Linear(in_features = 16*16*4, out_features = 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = self.cnn_layers(state)
        mu = self.linear_layers1(x)
        log_std = self.linear_layers2(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)
        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        #state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mu, log_std = self.forward(state.unsqueeze(0))
        std = log_std.exp()
        
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers

        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cnn_layers = nn.Sequential(
            # 定义2D卷积层
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 定义另一个2D卷积层
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.linear_layers1 = nn.Sequential(
            nn.Linear(16 * 64 * 1 + 2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.cnn_layers(state)
        x = torch.cat((x, action), dim=1)
        x = self.linear_layers1(x)
        return x

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, action_prior="uniform"):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR) 
        self._action_prior = action_prior
        
        print("Using: ", device)
        
        # Actor Network 
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)     
        
        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed).to(device)
        
        self.critic1_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) 

        self.save_dict = {"actor": self.actor_local, "critic1":self.critic1, "critic2":self.critic2, 
                                "critic1_target":self.critic1_target, "critic2_target":self.critic2_target}

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memory = PrioritizedReplay(capacity=BUFFER_SIZE)
    def add_sample(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.push(state, action, reward, next_state, done)


    def step(self, c_k):
        # Learn, if enough samples are available in memory
        experiences = self.memory.sample(BATCH_SIZE, c_k)
        self.learn( experiences, GAMMA)
            
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        img_state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_action(img_state).detach()
        return action

    def learn(self, experiences, gamma):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idx, weights = experiences
        states      = torch.FloatTensor(np.float32(states)).to(device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(device)
        actions = list(actions)
        for i in range(len(actions)) :
            actions[i] = actions[i].unsqueeze(0)
        actions     = torch.cat(actions, dim=0).to(device)#.unsqueeze(1)

        rewards     = torch.FloatTensor(rewards).to(device).unsqueeze(1) 
        dones       = torch.FloatTensor(dones).to(device).unsqueeze(1)
        weights    = torch.FloatTensor(weights).unsqueeze(1)
        #print(actions.shape)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next).cpu()

        # Compute Q targets for current states (y_i)
        Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))
        #TD_L1 = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target1_next.cpu() - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))
        #TD_L2 = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target2_next.cpu() - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))
        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        td_error1 = Q_targets.detach()-Q_1#,reduction="none"
        td_error2 = Q_targets.detach()-Q_2
        critic1_loss = 0.5* (td_error1.pow(2)*weights).mean()
        critic2_loss = 0.5* (td_error2.pow(2)*weights).mean()
        prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.memory.update_priorities(idx, prios.data.cpu().numpy())

        alpha = torch.exp(self.log_alpha)
        # Compute alpha loss
        actions_pred, log_pis = self.actor_local.evaluate(states)
        alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = alpha
        # Compute actor loss
        if self._action_prior == "normal":
            policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
            policy_prior_log_probs = policy_prior.log_prob(actions_pred)
        elif self._action_prior == "uniform":
            policy_prior_log_probs = 0.0
   
        actor_loss = ((alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs )*weights).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target, TAU)
        self.soft_update(self.critic2, self.critic2_target, TAU)
                     

    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, name):
        state = {key: value.state_dict() for key, value in self.save_dict.items()}
        torch.save(state, name)

    def load(self, name):
        checkpoint = torch.load(name)
        for model_name, model in self.save_dict.items():
            model.load_state_dict(checkpoint[model_name])

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, alpha=0.6, beta_start = 0.4, beta_frames=int(1e5)):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = deque(maxlen=capacity)
        self.pos        = 0
        self.priorities = deque(maxlen=capacity)
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of
        learning. In practice, we linearly anneal 
        from its initial value 
        0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        # assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        max_prio = max(self.priorities) if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        self.buffer.insert(0, (state, action, reward, next_state, done))
        self.priorities.insert(0, max_prio)
    
    
    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if c_k > N:
            c_k = N
            

        if N == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(list(self.priorities)[:c_k])
        
        #(prios)
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p and the c_k range of the buffer
        indices = np.random.choice(c_k, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        np.concatenate(states)
        np.concatenate(next_states)
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)


def SAC(n_interactions, save_every=100):
    
    csvUtil = CsvUtil(path = "draw.csv")

    scores_deque = deque(maxlen=args.print_every)
    state = env.reset()
    episode_K = 0
    eps_reward = 0
    eta_0 = 0.996
    eta_T = 1.0
    episodes = 0
    max_ep_len = 500 # original = 1000
    c_k_min = 2500 # original = 5000
    t = 0   
    #for t in range(1, int(n_interactions)+1):

    # agent.load("./data/5850.pt")

    while True:
        t +=1 
        action = agent.act(state)
        action_v = action.numpy()
        action_v = np.clip((action_v+action_bias)*action_scale, action_low, action_high)
        next_state, reward, done, info = env.step(action_v)
        agent.add_sample(state, action, reward, next_state, done)
        eta_t = eta_0 + (eta_T - eta_0)*(t/n_interactions)
        state = next_state
        eps_reward += reward
        episode_K +=1
        if done:
            episodes += 1
            if (args.test == False):
                for k in range(1,episode_K):
                    c_k = max(int(agent.memory.__len__()*eta_t**(k*(max_ep_len/episode_K))), c_k_min)
                    agent.step(c_k)
            
            scores_deque.append(eps_reward)
            writer.add_scalar("Reward", eps_reward, episodes)
            writer.add_scalar("average_X", np.mean(scores_deque), episodes)
            print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(episodes, eps_reward, np.mean(scores_deque)), end="")
            if episodes % save_every == 0:
                    agent.save("data/" + str(episodes) + ".pt")

            csvUtil.write(eps_reward)
            state = env.reset()
            episode_K = 0
            eps_reward = 0
           
    torch.save(agent.actor_local.state_dict(), args.info + ".pt")
        
        
def play():
    agent.actor_local.eval()
    for i_episode in range(1):
        state = env.reset()
      
        while True:
            action = agent.act(state)
            action_v = action[0].numpy()
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            next_state = next_state
            state = next_state

            if done:
                break 
    





parser = argparse.ArgumentParser()
parser.add_argument("-env", type=str, default="Pendulum-v0", help="Name of the Environment")
parser.add_argument("-frames", type=int, default=2000000, help="Number of frames to train, default = 20000")
parser.add_argument("-bs", "--buffer_size", type=int, default=int(1e6), help="Size of the Replay buffer, default= 1e6")
parser.add_argument("-bsize", "--batch_size", type=int, default=256, help="Batch size for the optimization process, default = 256")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=5e-4, help="Learning Rate, default 5e-4")
parser.add_argument("-g", type=float, default=0.99, help="discount factor gamma, default = 0.99")
parser.add_argument("-wd", type=float, default=0, help="Weight decay, default = 0")
parser.add_argument("-ls", "--layer_size", type=int, default=256, help="Number of nodes per neural network layer, default = 256")
parser.add_argument("--print_every", type=int, default=100, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-info", type=str, default="test", help="tensorboard test run information")
parser.add_argument("-device", type=str, default="cuda:0", help="Change to CPU computing or GPU, default=cuda:0")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("--save_freq", type=str, default=50, help="Load a saved model to perform a test run!")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-test", type=bool, default=False, help="Softupdate factor tau, default is 1e-2")

args = parser.parse_args()


if __name__ == "__main__":
    
    if args.test == True:
        print("**********   Testing Now!    ***********")        

    seed = args.seed
    BUFFER_SIZE = args.buffer_size
    BATCH_SIZE = args.batch_size        # minibatch size
    n_interactions = args.frames
    GAMMA = args.g            # discount factor
    TAU = args.tau            # for soft update of target parameters
    LR_ACTOR = args.lr         # learning rate of the actor 
    LR_CRITIC = args.lr       # learning rate of the critic
    WEIGHT_DECAY = args.wd#1e-2        # L2 weight decay
    HIDDEN_SIZE = args.layer_size
    saved_model = args.saved_model
    
    env_name = args.env
    device = args.device
    env = drone_env_heightcontrol(scaling_factor=5, start = [0,0,-4])
    np.random.seed(seed)
    state_size = 1
    action_size = 1
    action_scale = [1.0, 2.5/2]
    action_bias = [0.0, 1.0]
    action_high = [1.0, 2.5]
    action_low = [-1.0, 1.0]
    writer = SummaryWriter("runs/"+args.info)
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed, action_prior="uniform") #"normal"

    start_time = time.time()

    SAC(n_interactions=args.frames, save_every=args.save_freq)

    end_time = time.time()
    print("Training took: {} min".format((end_time-start_time)/60))
    #writer.add_hparams()
