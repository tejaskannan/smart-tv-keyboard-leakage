package smarttvsearch.suboptimal;

import smarttvsearch.utils.Move;

public class SuboptimalMoveFactory {

    public static SuboptimalMoveModel make(String name, Move[] moveSeq) {
        if (name.equals("standard")) {
            return new SuboptimalMoveModel(moveSeq);
        } else if (name.equals("credit_card")) {
            return new CreditCardMoveModel(moveSeq, 0.0,  0.0);
        } else {
            throw new IllegalArgumentException("Unknown move model with name: " + name);
        }
    }

}
