package SingletonImp;

/*
pros:lazy loading
cons:do not support multithreading
 */
public class Singleton1 {
  private static Singleton1 instance;
  private Singleton1(){ }

  public static Singleton1 getInstance(){
    if (instance==null)
      instance  = new Singleton1();
    return instance;
  }

  public void showMessage(){
    System.out.println("hello singleton");
  }
}
