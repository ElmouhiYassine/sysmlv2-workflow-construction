package PackageName {

    action def ProcessName {


        first start;

        then action action1 {
            description = "action1.";
        }
        then fork; then action2; then action5;
        action action2 {
            description = "action2.";
        }
        then fork; then action3; then action4;
        action action3 {
            description = "action3.";
        }
        then joinNode8;
        then joinNode4;
        action action5 {
            description = "action5.";
        }
        then fork; then action4; then action6;
        then joinNode4;
        join  joinNode4;
        then action action4 {
            description = "action3.";
        }
        then joinNode7;
        action action6 {
            description = "action6.";
        }
        then joinNode7;
        join  joinNode7;
        then action action7 {
            description = "action7.";
        }
        then joinNode8;
        join  joinNode8;
        then action action8 {
            description = "action8.";
        }

        then done;
    }
}
