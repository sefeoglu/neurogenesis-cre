

rand_seed = 16

np.random.seed(rand_seed) #for reproducibility

def get_phi(m,D, which_phi = 'performer'):

  '''Function that returns the random feature map, phi.
  Since our neuron-astrocyte model is equivalent to using Random Feature Attention, we use this representation for simplicity.
  Different phi functions lead to different feature maps.'''

  #random weight matrix for random feature map
  W = np.random.normal(0,1,(m,D))

  if which_phi == 'cosine':
    #random biases for cosine feature map
    rand_b = np.random.uniform(0,2*np.pi,m)

    def phi(x,c = 0):
      '''Uses an cosine  random feature map to approximate softmax attention.'''
      return np.sqrt(2/m)*np.cos(W @ x + rand_b)*np.exp(0.5*(np.linalg.norm(x)**2) - c)


  if which_phi == 'performer':
    def phi(x,c = 0):
      '''Uses an exponential random feature map to approximate softmax attention.'''
      #x has dimensions of the encoded tokens (for albert this is m = 768)
      return np.exp( -0.5*np.log(m) + W @ x - 0.5*(np.linalg.norm(x)**2))

  if which_phi == 'linear':
    def phi(x,c = 0):
      '''Uses an exponential random feature map to approximate softmax attention.'''
      #x has dimensions of the encoded tokens (for albert this is m = 768)
      h =  -0.5*np.log(m) + W @ x - 0.5*(np.linalg.norm(x)**2)
      return 1 + h

  if which_phi == 'truncated_performer':
    def phi(x,thresh = 150):
      '''Uses an exponential random feature map to approximate softmax attention.'''
      #x has dimensions of the encoded tokens (for albert this is m = 768)
      scaling_factors = np.exp( -0.5*np.log(m) - 0.5*(np.linalg.norm(x)**2))
      h = np.exp(W @ x)
      return scaling_factors*np.maximum(0,np.minimum(h,thresh))

  if which_phi == 'positive_cosine':
    #random biases for cosine feature map
    rand_b = np.random.uniform(0,2*np.pi,m)
    def phi(x,thresh = 10):
      '''Uses an exponential random feature map to approximate softmax attention.'''
      #x has dimensions of the encoded tokens (for albert this is m = 768)
      scaling_factors = np.sqrt(2/(np.pi*m))*np.exp(0.5*(np.linalg.norm(x)**2))
      h = np.cos(W @ x + rand_b)
      return np.maximum(0,scaling_factors*h)

  if which_phi == 'dima_sin':
    #random biases for cosine feature map
    rand_b = np.random.uniform(0,2*np.pi,m)

    def clipped_sin(x):
      if -np.pi/2 < x < np.pi/2:
        y = np.sin(x)
      if x > np.pi/2:
        y = 1
      if x < -np.pi/2:
        y = -1

      return y

    v_clipped_sin = np.vectorize(clipped_sin)


    def phi(x,thresh = 10):
      '''Uses an exponential random feature map to approximate softmax attention.'''
      #x has dimensions of the encoded tokens (for albert this is m = 768)
      scaling_factors = np.sqrt(2/m)*np.exp(0.5*(np.linalg.norm(x)**2))
      h = v_clipped_sin(W @ x + rand_b)
      return scaling_factors*h #np.maximum(0,scaling_factors*h)

  return phi


def get_astro_responses(query_layer,key_layer,nhead,phi):
  '''Computes astrocyte response given a random feature map, queries, and keys.'''

  ntokens = query_layer.shape[2]

  rfa_key_sum = 0
  for i in range(ntokens):
    rfa_normalized_keys = phi(key_layer[0,nhead,i,:])
    rfa_key_sum += rfa_normalized_keys

  astro_ps = np.zeros(ntokens)

  for t in range(ntokens):
    q_t = query_layer[0,nhead,t]
    astro_ps[t] = np.dot(phi(q_t),rfa_key_sum)

  return astro_ps
