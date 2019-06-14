#### Improvise a Jazz Solo with an LSTM Network

* keras layer objects

  1. Reshape()
  2. LSTM()
  3. Dense()
  4. Lambda()

* how to use keras model

  ```python
  def djmodel(Tx, n_a, n_values):
      
      # Define the input of your model with a shape 
      X = Input(shape=(Tx, n_values))
      
      # Define s0, initial hidden state for the decoder LSTM
      a0 = Input(shape=(n_a,), name='a0')
      c0 = Input(shape=(n_a,), name='c0')
      a = a0
      c = c0
       
      # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
      outputs = []
      
      # Step 2: Loop
      for t in range(Tx):
          
          # Step 2.A: select the "t"th time step vector from X. 
          x =  Lambda(lambda x: X[:,t,:])(X)
          # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
          x = reshapor(x)
          # Step 2.C: Perform one step of the LSTM_cell
          a, _, c = LSTM_cell(x, initial_state=[a, c])
          # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
          out = densor(c)
          # Step 2.E: add the output to "outputs"
          outputs.append(out)
          
      # Step 3: Create model instance
      model = Model(inputs=[X, a0, c0], outputs=outputs)
      
      return model
  ```

  ```python
  model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
  opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  ```

  ```python
  m = 60
  a0 = np.zeros((m, n_a))
  c0 = np.zeros((m, n_a))
  ```

  ```py
  model.fit([X, a0, c0], list(Y), epochs=100)
  ```

  ```python
  pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
  ```

* `keras.util.to_categorical()`

  ```python
  Convert indices to one-hot vectors, the shape of the results should be (1, )
  results = to_categorical(indices, num_classes=x_initializer.shape[2])
  ```

  