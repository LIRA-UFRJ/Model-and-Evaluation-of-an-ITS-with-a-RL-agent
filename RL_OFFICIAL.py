
# rl modelo final com 3 tipos de questões
import numpy as np
import matplotlib.pyplot as plt 
import random as rd
import pandas as pd
from scipy.optimize import minimize_scalar
import itertools
import time
import os
import matplotlib.ticker as mticker
import tqdm
import json

# para essa primeira modelagem iremos nos basear somente na quantidade no dado de quantidade de acertos e quantidade de questões feitas 

#global_lambdas = [0.1,0.05]  # lf , lvf
#global_lambdas = [0.025,0.0125]  # lf , lvf

global_lambdas = [0.05,0.025] # lf , lvf


max_episode_length = 12
min_episode_length = 0

# alfa = 1 # parametro da recompensa (R = I(acerto) * d^(1/alfa))

class Sistema:

# diminuir para três ou quatro tipos de questões, mudar semantica do max size episode,

    def __init__(self, k=0, df=1, policy = None, lambdas = global_lambdas, epsilon = None,alfa=1):
        

        self.epsilon = epsilon
        ## TEMPORÀRIO: Talvez não precise desse +1 em "self.max_size_episode"
        self.max_size_episode = max_episode_length + 1 # duração máxima de um episódio
        self.question_size = 3 # quantos nívieis de "dificuldade" cada questão tem
        self.k = k  # determina a quantidade de tipos de aluno, onde cada tipo é um quantil da distribuição de h, nosso parâmetro de habilidade
        self.alfa = alfa
    
        # initially every action has equal probability, soft policy, 12 dimensions considerando que no máximo uma sessão tem duração de max_size_episode timesteps e no máximo o episódio pode ter max_size_episode questões com também max_size quantidade de questões acertadas
        #s = { {a_j, q_j}_{j=1}, h}, as cinco primeiras tuplas contêm a quantidade de questões de um tipo acertadase e em uma sessão e a quantidade de questões feitas desse tipo e h é a haiblidade estimada do aluno
        if policy is None:
            if k==0:
            
                self.policy = np.full((self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.question_size),
                                   0.3335, dtype=np.float16)
            
            if k > 0:
                self.policy = np.full((self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,self.k,
                                   self.question_size),
                                   0.3335, dtype=np.float16)
        else: self.policy = policy 

        # to apply monte carlo algorithms we need the q for every pair state action, we initiallize it to 0 arbitrary. We also store the amount of times that a pair state action was experienced
        if k == 0:
            self.q_values = np.zeros((self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.question_size),)
        if k > 0:
            self.q_values = np.zeros((self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,self.k,
                                   self.question_size),)
            
        if k == 0:
            self.number_of_times_saw_q = np.zeros((self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.question_size), dtype=np.float16)
            
        if k > 0:
            self.number_of_times_saw_q = np.zeros((self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,
                                   self.max_size_episode, self.max_size_episode,self.k,
                                   self.question_size), dtype=np.float16)
            
        
        # discount factor
        self.discount_factor = df #por padrão utilizamos 1 pois episódios não são muito longos então podemos apenas dar peso igual a cada time step

        self.lambdas = lambdas #lambdas que são usados para o aluno

    def get_action(self, state, student, policy = None):
        """
        OBS:
            - Policy=None usa a política do objeto "Sistema", i.e., 'self.policy'.
            Depois de rodar self.op_ev_mc_control_sp, essa sera a política ótima.
        
        """
        
        if policy is None: # ação de acordo com self.policy
            p = np.asarray(self.policy[state]).astype('float16')
            p = p / np.sum(p)
            return np.random.choice(self.question_size, p=p)
      
        else: # ação de acordo com a política passada
             p = np.asarray(policy[state]).astype('float16')
             p = p / np.sum(p)
             return np.random.choice(self.question_size, p=p)
         
    

  
    def session(self, student, policy=None):

        # realizamos uma simulação de sessão para um estudante student
        # o session history tem um array para a sequência de estados, um para a sequência de ações e um para a sequência de recompensas
        session_history = [[],[],[]]
        # resetamos a sessão para o aluno que sempre começa no estado de 0 questões acertadas e 0 questões erradas 
        
        
        real_h = student.qi
            
        discretized_h = min(int(real_h * self.k), self.k - 1)
        
        state = [0,0,
                 0,0,
                 0,0,
                 discretized_h]
        
        # state = [0,0,
        #          0,0,
        #          0,0
        #          ]
        
        
        student.ativo = 1

        while student.ativo == 1:
            
            question = self.get_action(tuple(state), student=student, policy=policy)
           
            nota, recompensa = student.oferecer_questao(question)

            # usamos state[:] para fazermos uma cópia do estado
            session_history[0].append(state[:])
            session_history[1].append(question)
            session_history[2].append(recompensa)
          

            # incrementamos a quantidade de questões feitas e a quantidade de questões acertadas
            state[(2 * question)+1]+=1
            
            if(nota == 1): state[2*question]+=1


        
        #ao aluno sair da sessão temos que adicionar o último estado visitado 
        session_history[0].append(state[:])    
        
        # após uma sessão devemos resetar os parametros de erros e acertos de um aluno
        student.reset()

        return session_history
    
    
    def op_ev_mc_control_sp(self, student_count, session_count, policy = None, eps_greedy=None, retornar_pol_opt = False, path_to_save=None): #On policy every visit monte carlo control soft policies, for estimating pi as approx opt pi 
        
        #start = time.perf_counter()
        if self.epsilon is None: eps_greedy = 0.3
        else: eps_greedy = self.epsilon
        if policy is None: policy =self.policy
        else: policy = policy
        
        q_values = self.q_values
        number_of_times_saw_q = self.number_of_times_saw_q
        
        # aqui amostramos um aluno
        #h_range = np.linspace(0, 1,student_count)
        h_range = np.random.uniform(low=0,high=1,size=student_count)
        
        for i in tqdm.tqdm(range(student_count), desc="Training on Sampled Students", unit="student"):
            #h = np.random.uniform(low=0,high=1,size=1)
            h = h_range[i]
            student = Aluno(lambdas= self.lambdas,qi=h,alfa=self.alfa)
            # aqui para esse aluno extraimos uma certa quantidade de amostras
            
            c = 0
            while c < session_count:
                
                episode = self.session(student)
                g_return = 0
                cap_t = (len(episode[2]) - 1) 
                
                while cap_t >= 0:
                    g_return = (self.discount_factor * g_return) + episode[2][cap_t]
                    state_t = episode[0][cap_t]
                    action_t = episode[1][cap_t]

                    number_of_times_saw_q[tuple(state_t)][action_t]+=1
                    q_values[tuple(state_t)][action_t] = q_values[tuple(state_t)][action_t] + ((g_return - q_values[tuple(state_t)][action_t])/number_of_times_saw_q[tuple(state_t)][action_t])
                    
                    best_actions_index = np.where(q_values[tuple(state_t)]==np.max(q_values[tuple(state_t)]))
                    policy[tuple(state_t)]=np.array([(eps_greedy/self.question_size) for _ in range(self.question_size)]) #reset the values of the policy to that state so we improve it
                    policy[tuple(state_t)][best_actions_index] += (1-eps_greedy)/np.size(best_actions_index) 
                    cap_t-=1    
                c+=1
                # prosseguimos com o cálculo do nosso novo mle
              
        # aqui obtemos a aproximação de nossa política ótima e salvamos ela em memória de acordo com um path
        if retornar_pol_opt:
            # Reshape policy to a 2D array if it's not already,
            # where rows are states and columns are actions.
            # This makes it easier to apply vectorized operations.
            original_shape = policy.shape
            policy_reshaped = policy.reshape(-1, self.question_size)

            # Find the maximum value along the action dimension for each state
            max_values = np.max(policy_reshaped, axis=1, keepdims=True)

            # Create a boolean mask where values are equal to the max
            is_best_action = (policy_reshaped == max_values)

            # Calculate the number of best actions for each state
            num_best_actions = np.sum(is_best_action, axis=1, keepdims=True)

            # Create the new policy: 1/num_best_actions for best actions, 0 otherwise
            # Handle cases where num_best_actions might be zero (though it shouldn't be for valid policies)
            new_policy_values = np.where(num_best_actions > 0, 1 / num_best_actions, 0)
            policy_reshaped = is_best_action * new_policy_values

            # Reshape back to the original policy shape
            policy = policy_reshaped.reshape(original_shape)
        
        self.policy = policy

        if path_to_save is not None:
            # Ensure the directory exists
            os.makedirs(path_to_save, exist_ok=True)
            filename = os.path.join(path_to_save, "optimal_policy.npy")
            np.save(filename, policy)
            print(f"Optimal policy saved to {filename}")

    

    def save_pol(self, p):
        path = f"myfilelol{p}.txt"
        with open(path, "w") as f:
            shape_prefix = self.policy.shape[:-1]  # all but last dimension

            for idx in itertools.product(*[range(d) for d in shape_prefix]):
                line = self.policy[idx]  # 1D array over last dimension
                f.write(str(line.tolist()) + "\n")
        return
    


    def quantidade_eps_para_pol_opt(self, eps_greedy=None, count =10000, path =r"results/quantidade_eps_para_pol_opt", session_by_st = 1 ): #On policy every visit monte carlo control soft policies, for estimating pi as approx opt pi 
        # Aqui vamos achar quantos episódios são necessários para acharmos uma boa aproximação para a política ótima aplicando o algoritmo de monte carlo toda vez para um aluno diferente
        if eps_greedy is None: eps_greedy= self.epsilon
        # else: eps_greedy= 0.9

        policy = np.full((self.max_size_episode, self.max_size_episode,
                               self.max_size_episode, self.max_size_episode,
                               self.max_size_episode, self.max_size_episode,
                               self.question_size,self.k),
                               0.3335, dtype=np.float16)

        q_values = np.zeros((self.max_size_episode, self.max_size_episode,
                               self.max_size_episode, self.max_size_episode,
                               self.max_size_episode, self.max_size_episode,
                               self.question_size,self.k),)
        
        number_of_times_saw_q = np.zeros((self.max_size_episode, self.max_size_episode,
                               self.max_size_episode, self.max_size_episode,
                               self.max_size_episode, self.max_size_episode,
                               self.question_size,self.k), dtype=np.float16)


        # como não sabemos o nível de habilidade média do aluno ao iniciar a sessão vamos estimar ele como estando na faixa média
        
        value_first_state = [q_values[0,0,0,0,0,0,0.5].copy()]  
        return_by_session = []
        st_level = 10000

        for i in range(count):
            
            if count == 1: 
                student = Aluno(lambdas= self.lambdas, qi=0.5,alfa=self.alfa) 
                st_level = student.qi
            else: 
                student = Aluno(lambdas=self.lambdas, alfa=self.alfa)
                st_level = student.qi

            for aa in range(session_by_st):
                episode = self.session(student, policy = policy)
                g_return = 0
                cap_t = (len(episode[2]) - 1) 
                while cap_t >= 0:
                    g_return = (self.discount_factor * g_return) + episode[2][cap_t]
                    state_t = episode[0][cap_t]
                    action_t = episode[1][cap_t]
                    number_of_times_saw_q[tuple(state_t)][action_t]+=1
                    q_values[tuple(state_t)][action_t] = q_values[tuple(state_t)][action_t] + ((g_return - q_values[tuple(state_t)][action_t])/number_of_times_saw_q[tuple(state_t)][action_t])
                    best_actions_index = np.where(q_values[tuple(state_t)]==np.max(q_values[tuple(state_t)]))
                    policy[tuple(state_t)]=np.array([(eps_greedy/self.question_size) for _ in range(self.question_size)]) #reset the values of the policy to that state so we improve it
                    policy[tuple(state_t)][best_actions_index] += (1-eps_greedy)/np.size(best_actions_index) 
                    cap_t-=1    
                
                value_first_state.append(q_values[0,0,0,0,0,0].copy())
                
                return_by_session.append(g_return) 
 
        value_first_state = np.array(value_first_state)
        return_by_session = np.array(return_by_session)

        # ema_spans = [10, 50, 200, 1000]  # you can change these as needed
        # colors = cm.get_cmap('tab10')  # or 'Set1', 'Dark2', etc.
        ema_half_lives = [10, 100, 1000]
        
        colors = plt.get_cmap('tab10') 
        fig, ax = plt.subplots(figsize=(6,6))
        
        x = np.linspace(start=0, stop=len(value_first_state),num=len(value_first_state))  # 0,1,2,... episodes

        for i in range(len(value_first_state[1])):
            #########
            ### Nesse loop for tinham várias linhas com ax.plots diferentes, tirei todas 
            #########
            
            base_color = colors(i % 10)  # choose a base color for the action
            
            ax.plot(x, value_first_state[:, i], label=f'Question Difficulty {i+1}', linewidth=1.5, color=base_color)
            
            series = pd.Series(value_first_state[:, i])
            
            ema = series.ewm(halflife=100, adjust=False).mean()
            ax.plot(x, ema, linestyle=':', linewidth=1,
            label=f'EMA(Half-life={100}) Question Difficulty {i+1}', color=base_color, alpha=0.4 + 0.15 * (ema_half_lives.index(100)))
            
        
        # ax.legend(fontsize='small')  # or 'x-small', or a number like fontsize=8
        ax.legend(ncol=3, fontsize='x-small', loc='upper center', bbox_to_anchor=(0.5, -0.15))
        ax.set_yscale('log')
        ax.set_xlabel('Amount of student sessions')
        ax.set_ylabel('Q-value component (log scale)')
        fig.tight_layout()

        fig.savefig(f'{path}/q_value_white_bg_eps_greedy={eps_greedy}_student_count={count}_session_count_by_st={session_by_st}.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'{path}/q_value_transparent_bg_eps_greedy={eps_greedy}_student_count={count}_session_count_by_st={session_by_st}.png', dpi=300, bbox_inches='tight', transparent=True)
        
        plt.close()

        fig_return, ax_return = plt.subplots(figsize=(6,6))
        base_color_return = colors(0)
        x2 = np.linspace(start=0, stop=len(return_by_session),num=len(return_by_session))  # 0,1,2,... episodes
        series_2 = pd.Series(return_by_session)
        ema2 = series_2.ewm(halflife=100,adjust=False).mean()
        ax_return.plot(x2,ema2,linestyle=":", linewidth=1,label=f"EMA(Half-life={100})", color=base_color_return, alpha=1)
        ax_return.legend()
        ax_return.set_yscale('log')
        ax_return.set_xlabel('Amount of student sessions')
        ax_return.set_ylabel('Return of the session (log scale)')
        fig_return.tight_layout()

        fig_return.savefig(f'{path}/return_white_bg_eps_greedy={eps_greedy}_student_count={count}_session_count_by_st={session_by_st}_st_level={st_level}.png', dpi=300, bbox_inches='tight')
        fig_return.savefig(f'{path}/return_transparent_bg_eps_greedy={eps_greedy}_student_count={count}_session_count_by_st={session_by_st}_st_level={st_level}.png', dpi=300, bbox_inches='tight', transparent=True)
        
    
        plt.close()
        return



    def G_plots(self,G,T,questions_dist,policy_list,h_range):
        interval = [h_range[0],h_range[1]]
        figsize = (10,14)
        
        policy_names = ["Optimal" if policy is None else policy for policy in policy_list]    
        n_policies = len(policy_list)
        policy_colors = ["#6baed6", "#969696", "#08306b", "#31a354", "#fd8d3c", "#de2d26"]
        fig1,ax1 = plt.subplots(figsize=(6,6))
        fig2,ax2 = plt.subplots(figsize=(6,6))
        fig6,ax6 = plt.subplots(figsize=(6,6))
        
        fig3,ax3 = plt.subplots(figsize=(6,6))
        fig4,ax4 = plt.subplots(figsize=(6,6))
        fig5,ax5 = plt.subplots(figsize=(6,6))
        
        mean_return_over_h = [G[j][0].mean() for j in range(n_policies)]
        mean_duration_over_h = [T[j][0].mean() for j in range(n_policies)]
        
        df = pd.DataFrame({
            'Policy': policy_names,
            'Unconditional Mean Return': mean_return_over_h,
            'Unconditional Mean Duration': mean_duration_over_h
        })
        
        latex_table = df.to_latex(
            index=False,
            float_format="%.2f",
            caption="Policy Performance Comparison",
            label="tab:policy_performance",
            column_format="lrr"
        )
        
        print(latex_table)
        
        
        #fig4, ax4 = plt.subplots(figsize=(6,6))
        #fig5, ax5 = plt.subplots(figsize=(6,6))
        
        x = h_range
        
        ##########################################
        ### Plot de x = Habilidade e y = Retorno Médio para políticas NÃO BASELINE
        
        ax1.set_title(f"Average G per h")
        for j,policy in enumerate(policy_list[:-3]): # todas políticas menos as baselines 
            name = policy_names[j]
            color = policy_colors[j]
            # Plot the mean line
            line = ax1.plot(x, G[j][0], label=name,color=color)
            
            # Add shaded standard deviation region
            # ax1.fill_between(x, 
            #                  G[j][0] - G[j][1], 
            #                  G[j][0] + G[j][1], 
            #                  alpha=0.2, 
            #                  color=line[0].get_color())
            
        ax1.set_xlabel('Ability Level')
        ax1.set_ylabel('Average G values')
        ax1.legend()
        ax1.grid(alpha=0.25)

        ##########################################
        ### Plot de x = Habilidade e y = Retorno Médio para políticas BASELINE
        
        ax2.set_title(f"Average G per h")
        
        for j,policy in enumerate(policy_list[3:]): # políticas BASELINE
            i = j+3
            name = policy_names[i]
            color = policy_colors[i]
            # Plot the mean line
            line = ax2.plot(x, G[i][0], label=name,color=color)
            
            # Add shaded standard deviation region
            # ax2.fill_between(x, 
            #                  G[i][0] - G[i][1], 
            #                  G[i][0] + G[i][1], 
            #                  alpha=0.2, 
            #                  color=line[0].get_color())
           
        ax2.set_xlabel('Ability Level')
        ax2.set_ylabel('Average G values')
        ax2.legend()
        ax2.grid(alpha=0.25)
        
        ####################################################################
        ### Plot de x = Habilidade e y = Retorno Médio para TODAS políticas 
        
        ax6.set_title(f"Average G per h")
        
        for j,policy in enumerate(policy_list): # políticas BASELINE
            
            # controla quanto que cad acurva aparece no gráfico
            # as curvas baseline aparecem mais apagdas para facilitar a visualização
            if j < len(policy_list) - 3:
                alpha1 = 1
                alpha2 = 0.2
            
            else:
                alpha1 = 0.3
                alpha2 = 0.02
            
            name = policy_names[j]
            color = policy_colors[j]
            # Plot the mean line
            line = ax6.plot(x, G[j][0], label=name,alpha=alpha1,color=color)
            
            # Add shaded standard deviation region
            # ax6.fill_between(x, 
            #                  G[j][0] - G[j][1], 
            #                  G[j][0] + G[j][1], 
            #                  alpha=alpha2, 
            #                  color=line[0].get_color())
           
        ax6.set_xlabel('Ability Level')
        ax6.set_ylabel('Average G values')
        ax6.legend()
        ax6.grid(alpha=0.25)
        
        
        ###############################################
        # Plot de x = Habilidade e y = Tempo do Episódio Médio para políticas NÃO BASELINE
        ax3.set_title(f"Average Episode Duration per h")
        
        for j,policy in enumerate(policy_list[:-3]):
            if policy == None: name="Optimal"
            else: name = policy
            color = policy_colors[j]
            
            # Plot the mean line
            line = ax3.plot(x, T[j][0], label=name,color=color)
            ax3.scatter(x, T[j][0],color=color)
            
            # Add shaded standard deviation region
            
            # ax3.fill_between(x, 
            #                  T[j][0] - T[j][1], 
            #                  T[j][0] + T[j][1], 
            #                  alpha=0.2, 
            #                  color=line[0].get_color())
        
            
        
        ax3.set_xlabel('Ability Level')
        ax3.set_ylabel('Average Episode Duration')
        ax3.legend()
        ax3.grid(alpha=0.25)
        
        ############################################
        # Plot de x = Habilidade e y = Tempo do Episódio Médio para políticas BASELINE
        
        ax4.set_title(f"Average Episode Duration per h")
        
        for j,policy in enumerate(policy_list[3:]):
            if policy == None: name="Optimal"
            else: name = policy
            
            i = j+3
            color = policy_colors[i]
            
            # Plot the mean line
            line = ax4.plot(x, T[i][0], label=name,color=color)
            ax4.scatter(x, T[i][0],color=color)
            
            # Add shaded standard deviation region
            
            # ax4.fill_between(x, 
            #                  T[i][0] - T[i][1], 
            #                  T[i][0] + T[i][1], 
            #                  alpha=0.2, 
            #                  color=line[0].get_color())
        
            
        
        ax4.set_xlabel('Ability Level')
        ax4.set_ylabel('Average Episode Duration')
        ax4.legend()
        ax4.grid(alpha=0.25)


        ####################################################################
        ### Questions Distribution Histogram
        ax5.set_title(f"Questions Distribution Histogram")
        num_policies = len(policy_list[:-3])
        num_question_types = len(questions_dist[0])
        bar_width = 0.2
        index = np.arange(num_policies)

        policy_sums = questions_dist[:-3].sum(axis=1, keepdims=True)
        #print("ESSE È O SUM ", policy_sums)
        # Divide each count by the policy sum and multiply by 100 for percentage
        questions_dist_percentage = (questions_dist[:-3] / policy_sums) * 100 

        # Plot bars for each question type
        for i in range(num_question_types):
            difficulty_color = policy_colors[i+3]
            ax5.bar(index + i * bar_width, np.array(questions_dist_percentage)[:, i], bar_width, label=f'Question Difficullty {i+1}',color=difficulty_color)

        ax5.set_xlabel('Policy')
        ax5.set_ylabel('Relative Frequency of Questions')
        ax5.set_title(f'Amount of Questions by Type and Policy')
        ax5.set_xticks(index + bar_width)
        ax5.set_xticklabels(policy_names[:-3])
        ax5.legend()
        
        ax5.set_ylim(bottom=0, top=100) # Ensure y-axis goes up to 100%
        ax5.set_yticks(np.arange(0, 101, 20)) # This sets ticks at 0, 20, 40, 60, 80, 100
        
        ax5.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
        plt.tight_layout() # Adjust layout to prevent labels from overlappin

    
    def estimador_G_pi(self, base_policy = None, student_count =50,n_samples=1000, path=None,interval=[0,1],skip_baselines=False):
        alfa = self.alfa
        
        if base_policy is None: base_policy = self.policy
        else: base_policy = base_policy
        
        # como não sabemos o nível de habilidade média do aluno ao iniciar a sessão vamos estimar ele como estando na faixa média
        #mle = int((self.k/2)-1) 
        lower_h, upper_h = interval
        h_range = np.linspace(lower_h, upper_h,student_count)
        
        if skip_baselines == True:
            policy_list = ["Simple", "Random", None]
        else:     
            policy_list = ["Simple", "Random", None, "Easy", "Medium", "Hard"]
        
        policy_names = ["Optimal" if policy is None else policy for policy in policy_list]    
        #policy_colors =["lightblue", "Grey", "darkblue", "Green", "Orange", "Red"]
        #policy_colors = ["#1f78b4", "#a6a6a6", "#023858", "#33a02c", "#ff7f0e", "#e31a1c"]
        policy_colors = ["#6baed6", "#969696", "#08306b", "#31a354", "#fd8d3c", "#de2d26"]
        n_policies = len(policy_list)
        G = np.zeros((n_policies,2,student_count))
        T = np.zeros((n_policies,2,student_count))
        #duration = [[],[],[],[]]

        # questions_dist = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        questions_dist = np.array([np.zeros(self.question_size,) for i in range(n_policies)])
        
        

        for j,policy in enumerate(policy_list):
            name = policy_names[j]
            print(f"Evaluating {name} policy")
            
            G_pi = np.zeros((2,student_count))# array 2 x student count, onde a primeira linha é o RETORNO média e a segunda é o desvio padrão
            average_std_lenght_pi = np.zeros((2,student_count)) # array 2 x student count, onde a primeira linha é a DURAÇÃO média e a segunda é o desvio padrão
            
            for i in tqdm.tqdm(range(student_count), desc="Estimating G per student", unit="student"):
                # calcula para um aluno
                h = h_range[i]
                student = Aluno(qi=h,alfa=self.alfa)
                student_G = np.zeros(n_samples)
                student_ep_lenght = np.zeros(n_samples)
                for n in range(n_samples):
                    # calcula o G para uma trajetoria
                    # episode = self.session_para_G(student, policy)
                    episode = self.session_para_G(student,policy)
                    student_ep_lenght[n] = len(episode[2]) # guarda o tamanho do n-ésimo episódio

                    # OLHA AQUIIII
                    # duration[j].append(len(episode[0][0]))
                    #print("esse é o episódio ", episode)
                    # print("duração sessaõ : ",  len(episode[0]), " student ep lenght, ", len(episode[2]))

                    g_return = 0
                    cap_t = (len(episode[2]) - 1) 
                    while cap_t >= 0:
                        # calcula o G da trajetória
                        g_return = (self.discount_factor * g_return) + episode[2][cap_t]
                        questions_dist[j][episode[1][cap_t]]+=1
                        cap_t-=1    
                        
                    student_G[n] = g_return
                G_pi[0][i] = np.mean(student_G)
                G_pi[1][i] = np.std(student_G)
                # print(student_ep_lenght)
                average_std_lenght_pi[0][i] = np.mean(student_ep_lenght)
                average_std_lenght_pi[1][i] = np.std(student_ep_lenght)
            
            G[j] = G_pi
            T[j] = average_std_lenght_pi
            
                        # Gráfico
        return G,T,questions_dist, policy_list, h_range
        

        if path is not None:
            # fig1.savefig(f'{path}/retorno_politicas.png', dpi=300, bbox_inches='tight')
            # fig1.savefig(f'{path}/retorno_politicas_transparente.png', dpi=300, bbox_inches='tight', transparent=True)
            
            # fig2.savefig(f'{path}/duracao_sessao.png', dpi=300, bbox_inches='tight')
            # fig2.savefig(f'{path}/duracao_sessao_transparente.png', dpi=300, bbox_inches='tight', transparent=True)
            
            # fig3.savefig(f'{path}/hist_dificuldades.png', dpi=300, bbox_inches='tight')
            # fig3.savefig(f'{path}/hist_dificuldades_transparent_bg.png', dpi=300, bbox_inches='tight', transparent=True)
            
            # # fig4.savefig(f'{path}/SCATTER_SessaoMedia_white_bg.png', dpi=300, bbox_inches='tight')
            # # fig4.savefig(f'{path}/SCATTER_SessaoMedia_transparent_bg.png', dpi=300, bbox_inches='tight', transparent=True)
            
            # # fig5.savefig(f'{path}/HIST_SessaoMedia_white_bg.png', dpi=300, bbox_inches='tight')
            # # fig5.savefig(f'{path}/HIST_SessaoMedia_transparent_bg.png', dpi=300, bbox_inches='tight', transparent=True)
           _ = 1 + 1

        #plt.close()
        #return
    #########
    ### Bloco comentado com session_para_G alternativa
    #########

    def session_para_G(self, student,policy=None):

        # realizamos uma simulação de sessão para um estudante student
        # o session history tem uma array para a sequência de estados, uma para a sequência de ações e uma para a sequência de recompensas
        session_history = [[],[],[]]
        # resetamos a sessão para o aluno que sempre começa no estado de 0 questões acertadas e 0 questões erradas e com o tipo do estudante a princípio no nível médoo self.k/2 se não tiver feito nenhuma sessão antes
        
        real_h = student.qi
            
        discretized_h = min(int(real_h * self.k), self.k - 1)
        state = [0,0,
                 0,0,
                 0,0,
                 discretized_h]
        
        # state = [0,0,
        #          0,0,
        #          0,0
        #          ]
        
        
        student.ativo = 1
        counter = 0



        while student.ativo == 1:
            if counter == 0:
                prev_state = state
            if counter > 0:
                prev_state = session_history[0][counter - 1] # guarda o estado anterior ao atual para calcular a política simples ("Simple Plicy")
            
            question = self.get_action_para_G(tuple(state), tuple(prev_state), student, policy)
            
            nota, recompensa = student.oferecer_questao(question)

            # usamos state[:] para fazermos uma cópia do estado
            session_history[0].append(state[:])
            session_history[1].append(question)
            session_history[2].append(recompensa)

        
            # incrementamos a quantidade de questões feitas e a quantidade de questões acertadas
            counter += 1
            state[(2 * question)+1]+=1
            if(nota == 1): state[2*question]+=1

                

        
        #ao aluno sair da sessão temos que adicionar o último estado visitado e resetar as variaveis do aluno
        session_history[0].append(state[:])    
        student.reset()

        return session_history

    def get_action_para_G(self, state, previous_state, student, policy=None):
        # if policy == None and np.sum(self.policy[state]) == 1:
        #     return np.random.choice(self.question_size, p=self.policy[state])
        if policy == None:
            return np.random.choice(self.question_size, p=self.policy[state])
        
        if policy == "Random":
            return np.random.choice(self.question_size)
    
        if policy == "Simple":

            if student.acertos ==0 and student.erros == 0:
                return np.random.choice(self.question_size)
            else: 
                if student.acertou_ultima == 1:
                    if student.ultima == 2: return 2
                    else: return student.ultima+1
                if student.acertou_ultima == 0:
                    if student.ultima == 0: return 0
                    else: return student.ultima-1
                    
        if policy=="Easy":
            return 0
        if policy=="Medium":
            return 1
        if policy=="Hard":
            return 2
            #########
            ### Linhas que calculavam algum vetor de diferença entre as dificuldades
            #########
            
        
        else:
            print("Nehuma política válida, escolhendo a aleatória")
            return np.random.choice(self.question_size)








class Aluno():

    def __init__(self, qi=None,acertos=0,erros=0,status=1,ultima_questao=0, lambdas=global_lambdas, alfa=1):
        self.acertos = acertos
        self.erros = erros
        self.ativo = status
        self.ultima = ultima_questao
        self.acertou_ultima = 0
        #self.alphas = np.array([9, 5, 2, 0.75, 0.7]) 
        self.lambdas = lambdas
        self.alfa=alfa
        self.F0, self.V0 = 0,0


        self.a,self.b = 100,110 # parametros da beta
        if qi is None:
           self.qi = np.random.beta(self.a,self.b)
        else:
           self.qi = qi


    def summary(self):
        traducao = {0: "Não" ,1: "Sim"}
        print("Número Total de Acertos: %d" % self.acertos)
        print("Número Total de Erros: %d" % self.erros)
        print("Ainda está na sessão:  %s " %  traducao[self.ativo])
        print("Última questão: %d" % self.ultima)
        print("Acertou a última: %s " % traducao[self.acertou_ultima])



    # SHAPE  ORIGNAL ERA 15
    def prob_func(self,h,d,shape=15,question_size=3):
        
        """Divide o intervalo [0,1] em 'question_size' pontos igualmente espaçados e definine 
        uma sigmoide centrada em cada ponto. A probabilidade de acerto vai ser o valor de sigmoid(h), 
        onde o centro da sigmoide depende da dificulade da questão. """
        
        partition_size = 1 / (question_size + 1) # define tamanho de cada pedaço da partição que vai nos dar os midpoints das sigmoides
        sigmoid_mid_points = [partition_size * i for i in range(1,question_size+1)]
        x0 = sigmoid_mid_points[d] # dificuldade = 1, ..., question_size
        sigmoid = 1 /(1 + np.exp(-shape*(h-x0)))
        return sigmoid

    
    def oferecer_questao(self, dificuldade):
        # print("dificuldade ", dificuldade)
        
        lf, lvf = self.lambdas
        if self.ativo == 1:
        
            Pd = self.prob_func(h=self.qi, d=dificuldade)
            # Aluno responde a Questão:
            nota = np.random.binomial(1,Pd) # bernoulli com probabilidade Pd
            # print("nota", nota)
            if nota == 0: # aluno errou
                self.erros += 1
                self.acertou_ultima = 0
            if nota == 1: # aluno acertou
                self.acertos += 1 
                self.acertou_ultima = 1

            # Essa é registrada como a última questão feita
            self.ultima = dificuldade
            
            V = self.acertos
            F = self.erros
            
            
            # V = nota
            # F = nota
            
            #Ps = np.exp(-((lf * F) + (lvf * (V+F))))
            
            # print(self.erros+self.acertos<min_episdode_lenght)
            # if (self.erros+self.acertos<min_episdode_lenght):
            #     Ps = 1
            # if self.erros+self.acertos==min_episdode_lenght:
            #     F0 = F
            #     V0 = V
            #     F = F - F0
            #     V = V - V0
            #     Ps = min([np.exp(-lf * F),np.exp(-lvf * (V+F))])
                
            # if self.erros+self.acertos>min_episdode_lenght:
            #     F = F - F0
            #     V = V - V0
            #     Ps = min([np.exp(-lf * F),np.exp(-lvf * (V+F))])
            
            
            episode_len = self.erros + self.acertos
            #print(episode_len)
            if episode_len < min_episode_length:
                Ps = 1
            
            else:
                if episode_len == min_episode_length:
                    self.F0 = F
                    self.V0 = V
            
                F_adj = F - self.F0
                V_adj = V - self.V0
            
                Ps = min(
                    np.exp(-lf * F_adj),
                    np.exp(-lvf * (V_adj + F_adj))
                )
            
            # Aluno decide se vai continuar ou não
            # if Ps == 0.01: Ps = 0
            
            ## Limites para tamanho de sessão, estou comentando porque sipa é reduntante:
            
            if (self.erros+self.acertos==max_episode_length): 
                Ps = 0
            # if(self.erros==8): Ps= 0
            # if(self.acertos == 10): Ps=0
            continua = np.random.binomial(1,Ps)
            
            if continua == 0: # aluno saiu saiu do app
                self.ativo = 0
            if continua == 1: # aluno ainda esta no app
                pass
            
            # aqui fornecemos a informação se o aluno acertou e a recompensa que ele recebe
            #recompensa = nota * (dificuldade) ** (1/self.alfa) 
            d = dificuldade+1
            recompensa = (nota * d) 
            
            
            return nota, recompensa
                
        elif self.ativo == 0:
            pass
        else:
            print("Erro no aluno.ativo")

    def reset(self):
        # reseta a quantidade de erros e acertos de um aluno para ser utilizado em uma nova sessão
        self.erros = 0
        self.acertos = 0
        self.acertou_ultima= 0
        self.F0, self.V0 = 0,0
        
        return
    




Sn = 2000 # quantidade de estudantes sampleados no treinamento
N = 1000 # quantidade de sessões por estudante
eps = 0.7 # original_eps = 0.7
k = 0


global_alfa = 1

# definição dos parâmetros do modelo
sistema = Sistema(k=k, df= 1, lambdas=global_lambdas, epsilon=eps,alfa=global_alfa)

# treinamento
sistema.op_ev_mc_control_sp(student_count=Sn,session_count=N,eps_greedy=eps,retornar_pol_opt=True)

# plots
G,T,questions_dist, policy_list, h_range = sistema.estimador_G_pi(student_count=Sn, n_samples=N,path=r"C:\Users\Vitor\Códigos\IC\its\results",interval=[0,1]) 

data = {
    "G": G.tolist(),
    "T": T.tolist(),
    "questions_dist": questions_dist.tolist(),
    "policy_list": policy_list,
    "h_range": h_range.tolist()
}

with open(f"results_k{k}_min{min_episode_length}.json", "w") as f:
    json.dump(data, f)

sistema.G_plots(G,T, questions_dist, policy_list, h_range)


# # interval_list = [[0,0.5], [0.5,1],
# #                  [0,1/3], [1/3,2/3], [2/3,1],
# #                  [0,1/4], [1/4,2/4], [2/4,3/4], [3/4,1]]

# interval_list = [[0,1/4], [1/4,2/4], [2/4,3/4], [3/4,1]]


# for interval in interval_list:
#     sistema.estimador_G_pi(student_count=Sn, n_samples=N,path=r"C:\Users\Vitor\Códigos\IC\its\results",interval=interval) 


# sistema = Sistema(k= 2, df= 1, lambdas=global_lambdas, epsilon=0.7)
# sistema.estimador_G_pi(student_count=Sn, n_samples=N,path=r"C:\Users\Vitor\Códigos\IC\its\results",interval=[0,1],skip_baselines=True) 


index = ["Simple", "Random", "Optimal", "Easy", "Medium", "Hard"]
data = []

pd.Series()

#### Plot das sigmoides
fig,ax = plt.subplots()

H = np.linspace(0,1,1000)
for d in [0,1,2]:
    sigmoid_d = Aluno().prob_func(H, d) 
    ax.plot(H,sigmoid_d,label=f"Dificuldade {d}")


ax.set_xlabel("h")
ax.set_ylabel("$P_d$")

ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.grid(alpha=0.25)
ax.axvline(x=0.25,alpha=0.2,color="black")
ax.axvline(x=0.5,alpha=0.2,color="black")
ax.axvline(x=0.75,alpha=0.2,color="black")
plt.legend()
plt.tight_layout()
    

### Plot das exponenciais
from matplotlib.ticker import MaxNLocator

I = np.linspace(0, 20, 200)

exp1 = np.exp(-global_lambdas[0] * I)
exp2 = np.exp(-global_lambdas[1] * I)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(I, exp1, 'o-', markersize=8, markevery=10)
ax1.set_xlabel('F')
ax1.set_ylabel(r'exp($-\lambda_1 F$)')
ax1.grid(True, alpha=0.3)

ax2.plot(I, exp2, 'o-', markersize=8, markevery=10)
ax2.set_xlabel('t')
ax2.set_ylabel(r'exp($-\lambda_2 t$)')
ax2.grid(True, alpha=0.3)

# Common y-range based on both curves
# ymin = min(exp1.min(), exp2.min())
# ymax = max(exp1.max(), exp2.max())

ymin = exp2.min()
ymax = exp2.max()


# Add a small margin so points don't touch the border
margin = 0.03 * (ymax - ymin)


for ax in (ax1, ax2):
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.yaxis.set_major_locator(MaxNLocator(6))


plt.tight_layout()
plt.show()

# sistema = Sistema(k= 2, df= 1, lambdas=global_lambdas, epsilon=0.7,alfa="new_reward")
# sistema.op_ev_mc_control_sp(student_count=Sn,session_count=N,eps_greedy=0.7,retornar_pol_opt=True)
# sistema.estimador_G_pi(student_count=Sn, n_samples=N,path=r"C:\Users\Vitor\Códigos\IC\its\results") 





# sistema.estimador_G_pi(student_count=500, n_samples=1000, path="/Users/peepoo/Documents/programming/IC/Artigo_Vitor_Fernando")

#########
### Bloco comentado com função que calcualva estatísticas para cada política
#########


#########
### Bloco comentado com um bando de calls de Sistema, op_ev_mc_control_sp e algumas observações
#########

