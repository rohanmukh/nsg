import java.*;

public class JavaCompiler {

   public static void main(String[] args) {
      boolean check = checker("java.util.List", "java.util.ArrayList");
   }

   public static boolean checker(String className1, String className2) {
      boolean check = false;
      Class class1, class2; 
      try {
          class1 = Class.forName(className1);
          class2 = Class.forName(className2);
          check = type_check(class1, class2);
      } catch (ClassNotFoundException e) {
          System.out.println(e);
      } catch (NullPointerException e) {
          System.out.println(e);
      }
      return check;
   }

   public static boolean type_check(Class c1, Class c2) throws NullPointerException
   {
           // your code goes here
           boolean sameType = c1.isAssignableFrom(c2);
           return sameType;
   }


}
