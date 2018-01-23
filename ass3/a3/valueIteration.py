import matplotlib.pyplot as plt

def value_iteration(S,small_number,p_h):
    #list for all value of state
    V = [0 for i in range(101)]
    optimal = [0 for i in range(100)]
    loop = True
    while loop:
        difference = 0
        for s in S:
            v = V[s]
            #list for all possible action
            action = [i+1 for i in range(min(s,100-s))]
            value = [0 for i in range(len(action)+1)]
            
            for a in action:
                if a+s>=100:
                    value[a] = p_h*(1+dis*V[s+a])+(1-p_h)*(0+dis*V[s-a])
                else:
                    value[a] = p_h*(0+dis*V[s+a])+(1-p_h)*(0+dis*V[s-a])
            #find the max value except V[0] and V[100]
            V[s] = max(value)
            optimal[s-1] = value.index(V[s])
            difference = max(difference,abs(v-V[s]))
        if difference<small_number:
            loop = False
            V[100] = 1 
        
    return(V,optimal)
                
             
                


if __name__=="__main__":
    #array for all the state
    S = [i for i in range(1,100)]
    small_number = 0.0001
    #probability of get head
    p_h = 0.25
    dis = 1
    (V,optimal)=value_iteration(S,small_number,p_h)
    
    
    
    
    #draw plot
    plt.show()
    plt.plot(V,label='')
    plt.xlim([0,100])
    plt.xticks([1,25,50,75,99])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend()
    plt.show()
    
    plt.show()
    plt.plot(optimal,label='')
    plt.xlim([0,100])
    plt.xticks([1,25,50,75,99])
    plt.xlabel('Capital')
    plt.ylabel('policy')
    plt.legend()
    plt.show()    
    
