# https://www.tcom242242.net/entry/ai-2/%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92/%E3%80%90%E6%B7%B1%E5%B1%A4%E5%BC%B7%E5%8C%96%E5%AD%A6%E7%BF%92%E3%80%91double-q-network/#toc9



class DDQNAgent():
    """
        Double Deep Q Network Agent
    """

    def __init__(self, training=None, policy=None, gamma=0.99, actions=None,
                memory=None, memory_interval=1, train_interval=1,
                batch_size=32, nb_steps_warmup=200,
                observation=None, input_shape=None, sess=None):

        self.training = training
        self.policy = policy
        self.actions = actions
        self.gamma = gamma
        self.recent_observation = observation
        self.previous_observation = observation
        self.memory = memory
        self.memory_interval = memory_interval
        self.batch_size = batch_size
        self.recent_action_id = None
        self.nb_steps_warmup = nb_steps_warmup
        self.sess = sess

        self.model_inputs, self.model_outputs, self.model = build_model(input_shape, len(self.actions))
        self.target_model_inputs, self.target_model_outputs, self.target_model = build_model(input_shape, len(self.actions))
        target_model_weights = self.target_model.trainable_weights
        model_weights = self.model.trainable_weights

        # hard update
        # self.update_target_model = [target_model_weights[i].assign(model_weights[i]) for i in range(len(target_model_weights))]
        # soft update
        self.update_target_model = [target_model_weights[i].assign(
            .999*target_model_weights[i]+.001*model_weights[i]) for i in range(len(target_model_weights))]
        self.train_interval = train_interval
        self.step = 0

    def compile(self):
        self.targets = tf.placeholder(dtype=tf.float32, shape=[
                                    None, 2], name="target_q")
        self.inputs = tf.placeholder(
            dtype=tf.int32, shape=[None], name="action")
        actions_one_hot = tf.one_hot(indices=self.inputs, depth=len(
            self.actions), on_value=1.0, off_value=0.0, name="action_one_hot")

        pred_q = tf.multiply(self.model_outputs, actions_one_hot)

        error = self.targets - pred_q
        square_error = .5 * tf.square(error)
        loss = tf.reduce_mean(square_error, axis=0, name="loss")

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train = optimizer.minimize(loss)
        self.sess.run(tf.initialize_all_variables())

    def act(self):
        action_id = self.forward()
        action = self.actions[action_id]
        return action

    def forward(self):
        q_values = self.compute_q_values(self.recent_observation)
        action_id = self.policy.select_action(
            q_values=q_values, is_training=self.training)
        self.recent_action_id = action_id

        return action_id

    def observe(self, observation, reward=None, is_terminal=None):
        self.previous_observation = copy.deepcopy(self.recent_observation)
        self.recent_observation = observation

        if self.training and reward is not None:
            if self.step % self.memory_interval == 0:
                self.memory.append(
                    self.previous_observation, self.recent_action_id, reward, terminal=is_terminal)
            self.experience_replay()
            self.policy.decay_eps_rate()

        self.step += 1

    def experience_replay(self):
        if (self.step > self.nb_steps_warmup) and (self.step % self.train_interval == 0):
            experiences = self.memory.sample(self.batch_size)

            state0_batch = []
            reward_batch = []
            action_batch = []
            state1_batch = []
            terminal_batch = []

            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal_batch.append(0. if e.terminal else 1.)

            target_batch = np.zeros((self.batch_size, len(self.actions)))
            reward_batch = np.array(reward_batch)

            # q values of q network
            q_values = self.predict_on_batch_by_model(state1_batch)
            # argmax actions of q network
            argmax_actions = np.argmax(q_values, axis=1)
            target_q_values = np.array(self.predict_on_batch_by_target(state1_batch))  # compute maxQ'(s')
            # Q(s', argmax_a Q(s, a;theta_q); theta_target), 
            double_q_values = []
            for a, t in zip(argmax_actions, target_q_values):
                double_q_values.append(t[a])
            double_q_values = np.array(double_q_values)

            discounted_reward_batch = (self.gamma * double_q_values)
            discounted_reward_batch *= terminal_batch
            # target = r + γ maxQ'(s')
            targets = reward_batch + discounted_reward_batch

            for idx, (action, target) in enumerate(zip(action_batch, targets)):
                target_batch[idx][action] = target

            self.train_on_batch(state0_batch, action_batch, target_batch)

        # soft update
        self.sess.run(self.update_target_model)

    def train_on_batch(self, state_batch, action_batch, targets):
        self.sess.run(self.train, feed_dict={
                        self.model_inputs: state_batch, self.inputs: action_batch, self.targets: targets})

    def compute_target_q_value(self, state1_batch):
        q_values = self.sess.run(self.target_model_outputs, feed_dict={
                                    self.target_model_inputs: state1_batch})
        q_values = np.max(q_values, axis=1)

        return q_values

    def predict_on_batch_by_model(self, state1_batch):
        q_values = self.sess.run(self.model_outputs, feed_dict={
                                self.model_inputs: state1_batch})

        return q_values

    def predict_on_batch_by_target(self, state1_batch):
        q_values = self.sess.run(self.target_model_outputs, feed_dict={
                                self.target_model_inputs: state1_batch})
        return q_values

    def compute_q_values(self, state):
        q_values = self.sess.run(self.target_model_outputs, feed_dict={
                                    self.target_model_inputs: [state]})

        return q_values[0]

    def update_target_model_hard(self):
        """ for hard update """
        self.sess.run(self.update_target_model)

    def reset(self):
        self.recent_observation = None
        self.previous_observation = None
        self.recent_action_id = None