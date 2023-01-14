package smarttvsearch;

import smarttvsearch.prior.LanguagePrior;
import smarttvsearch.prior.LanguagePriorFactory;
import smarttvsearch.keyboard.Keyboard;
import smarttvsearch.keyboard.MultiKeyboard;
import smarttvsearch.utils.Direction;
import smarttvsearch.utils.Move;
import smarttvsearch.utils.SmartTVType;
import smarttvsearch.utils.sounds.SmartTVSound;
import smarttvsearch.utils.sounds.SamsungSound;

public class Search {

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("Must provide a file path.");
            return;
        }

        //System.out.println(args[0]);
        //System.out.println(Direction.ANY.name());
        //Keyboard keyboard = new Keyboard(args[0]);
        MultiKeyboard keyboard = new MultiKeyboard(SmartTVType.SAMSUNG, args[0]);

        //SmartTVSound endSound = new SamsungSound("delete");
        //Move move = new Move(5, endSound);

        //System.out.println(keyboard.getNeighbors("v", false, false, Direction.ANY));
        //System.out.println(keyboard.getNeighbors("v", false, true, Direction.VERTICAL));

        //LanguagePrior prior = LanguagePriorFactory.makePrior("numeric", "");

        //LanguagePrior prior = new NGramPrior(args[0]);
        //prior.build(false);

        //System.out.println(prior.find("lakers"));
    }

}
