package smarttvsearch.tests.keyboard;

import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;


public class KeyboardLinkerTestRunner {

    public static void main(String[] args) {
        Result result = JUnitCore.runClasses(KeyboardLinkerTests.class);

        for (Failure failure : result.getFailures()) {
            System.out.println("Failure: " + failure.toString());
        }

        if (result.wasSuccessful()) {
            System.out.printf("Passed all %d tests.\n", result.getRunCount());
        } else {
            System.out.printf("Failed %d / %d tests.\n", result.getFailureCount(), result.getRunCount());
        }
    }

}
