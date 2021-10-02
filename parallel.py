import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle

env = retro.make("SuperMarioBros3-Nes", '1Player.World1.Level8')
class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        ob = env.reset()

        env.action_space.sample()
        inx, iny, _ = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)
        done = False

        net = neat.nn.RecurrentNetwork.create(self.genome, self.config)


        fitness = 0
        curr_x = 0
        curr_score = 0
        counter = 0
        imgarray = []

        while not done:
            #env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)
            actions = net.activate(imgarray)
            ob, _, done, info = env.step(actions)

            x = info['x']
            score = info['score']
            live = info['lives']
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

        #env.reset()
        print(fitness)
        return fitness

def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

p = neat.Population(config)
# p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-161')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(8, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

