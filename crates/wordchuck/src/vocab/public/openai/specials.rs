//! # Special Tokens

use crate::declare_carrot_special;

declare_carrot_special!(
    (STARTOFTEXT, "startoftext"),
    (ENDOFTEXT, "endoftext"),
    (ENDOFPROMPT, "endofprompt"),
    (FIM_PREFIX, "fim_prefix"),
    (FIM_MIDDLE, "fim_middle"),
    (FIM_SUFFIX, "fim_suffix"),
    (RETURN, "return"),
    (CONSTRAIN, "constrain"),
    (CHANNEL, "channel"),
    (START, "start"),
    (END, "end"),
    (MESSAGE, "message"),
    (CALL, "call"),
);
