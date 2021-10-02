import retro
import numpy as np
import cv2
import neat
import pickle

# env = retro.make('SuperMarioBros3-Nes')
env = retro.make('SuperMarioBros3-Nes', '1Player.World1.Level1.1')

def eval_genomes(genomes, config):
    imgarray = []
    for genome_id, genome in genomes:
        ob = env.reset()
        env.action_space.sample()

        inx, iny, _ = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)
        
        net = neat.nn.RecurrentNetwork.create(genome, config)
        done = False

        fitness = 0
        curr_x = 0
        curr_score = 0
        counter = 0
        imgarray = []

        while not done:
            env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            actions = net.activate(imgarray)
            ob, _, done, info = env.step(actions)

            x = info['x']
            score = info['score']
            live = info['lives']
            # print(x)
            if score > curr_score:
                curr_score = score
                counter = 0
                fitness += 10
            if x > curr_x:
                curr_x = x
                counter = 0
                fitness += 100
            if live < 4:
                done = True
            else:
                counter += 1
                
            if counter > 250:
                done = True
            if x == 160:
                fitness += 100000
                done = True
            genome.fitness = fitness
        print(genome_id, fitness)

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

p = neat.Population(config)

p = neat.Checkpointer.restore_checkpoint('checkpoint-level1')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
