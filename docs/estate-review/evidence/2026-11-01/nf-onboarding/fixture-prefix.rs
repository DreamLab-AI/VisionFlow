#![allow(dead_code,unused_variables)]
use std::cell::RefCell;
#[derive(Clone, PartialEq, Debug)]
enum ConnectState {Working,Success,Error(String)}
#[derive(Clone)]
struct World { window:bool, history:bool, replace_ok:bool, print_ok:bool, hash:String, trace:Vec<&'static str>, result:ConnectState, printed:bool, confirmed:bool, ready:bool, timeout:Option<i32>, imported:Option<String> }
impl Default for World {fn default()->Self{Self{window:true,history:true,replace_ok:true,print_ok:true,hash:String::new(),trace:vec![],result:ConnectState::Working,printed:false,confirmed:false,ready:false,timeout:None,imported:None}}}
thread_local!{static WORLD:RefCell<World>=RefCell::new(World::default());}
fn reset(hash:&str){WORLD.with(|w|*w.borrow_mut()=World{hash:hash.into(),..World::default()});}
fn snapshot()->World{WORLD.with(|w|w.borrow().clone())}
fn base_href(path:&str)->String{format!("/community{path}")}
fn current_app_path(path:&str)->String{base_href(path)}
mod wasm_bindgen {pub struct JsValue;impl JsValue{pub const NULL:Self=Self;}}
mod web_sys {
 use super::*;
 pub struct Window;pub struct Location;pub struct History;
 pub fn window()->Option<Window>{WORLD.with(|w|if w.borrow().window{Some(Window)}else{None})}
 impl Window{pub fn location(&self)->Location{Location} pub fn history(&self)->Result<History,()>{WORLD.with(|w|if w.borrow().history{Ok(History)}else{Err(())})} pub fn print(&self)->Result<(),()>{WORLD.with(|w|{let mut w=w.borrow_mut();w.trace.push("print");if w.print_ok{Ok(())}else{Err(())}})}}
 impl Location{pub fn hash(&self)->Result<String,()>{WORLD.with(|w|{let mut w=w.borrow_mut();w.trace.push("read");Ok(w.hash.clone())})}}
 impl History{pub fn replace_state_with_url(&self,_:&wasm_bindgen::JsValue,_:&str,_:Option<&str>)->Result<(),()>{WORLD.with(|w|{let mut w=w.borrow_mut();w.trace.push("strip");if w.replace_ok{w.hash.clear();Ok(())}else{Err(())}})}}
}
#[derive(Clone,Copy)]struct Auth;impl Auth{fn clear_error(&self){} fn login_with_local_key(&self,k:&str)->Result<(),String>{WORLD.with(|w|w.borrow_mut().trace.push("import"));let p=validate_key(k)?;WORLD.with(|w|w.borrow_mut().imported=Some(p));Ok(())}}
#[derive(Clone,Copy)]struct State;impl State{fn set(&self,s:ConnectState){WORLD.with(|w|w.borrow_mut().result=s)}}
#[derive(Clone,Copy)]struct Navigate;impl Navigate{fn with_value<F:FnOnce(&fn(&str,NavigateOptions))>(&self,f:F){let n:fn(&str,NavigateOptions)=|_,_|{};f(&n)}}
#[derive(Default)]struct NavigateOptions;
fn set_timeout_once<F:FnOnce()+'static>(_:F,ms:i32){WORLD.with(|w|w.borrow_mut().timeout=Some(ms));}
struct Printed;impl Printed{fn get(&self)->bool{snapshot().printed}fn set(&self,v:bool){WORLD.with(|w|w.borrow_mut().printed=v)}}
struct Confirmed;impl Confirmed{fn get(&self)->bool{snapshot().confirmed}}
struct Ready;impl Ready{fn run(&self,_:()){WORLD.with(|w|w.borrow_mut().ready=true)}}
