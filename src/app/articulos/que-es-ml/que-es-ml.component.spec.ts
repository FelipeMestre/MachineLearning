import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QueEsMLComponent } from './que-es-ml.component';

describe('QueEsMLComponent', () => {
  let component: QueEsMLComponent;
  let fixture: ComponentFixture<QueEsMLComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ QueEsMLComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(QueEsMLComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
