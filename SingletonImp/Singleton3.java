package SingletonImp;

/*
pros:multiple threads supporting, effective
cons:not lazy loading,waste memory
 */
public class Singleton3 {
  private static Singleton3 instance = new Singleton3();
  private Singleton3(){}
  public static Singleton3 getInstance(){
    return instance;
  }
}
